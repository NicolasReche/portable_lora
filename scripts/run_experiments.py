"""
run_experiments.py

Master script that orchestrates the full experimental pipeline for the thesis:

    PHASE 1 — Train RL LoRA modules (one per attribute per dataset per seed)
    PHASE 2 — Evaluate single modules + all composition techniques
    PHASE 3 — Port modules to a new backbone, evaluate at 0 fine-tuning steps
    PHASE 4 — (Optional) Evaluate composition of ported modules

This replicates and extends your previous paper's experimental structure
(Table 1), now with RL-trained modules and the portability dimension added.

Usage:
    # Run everything
    python run_experiments.py --config configs/full_experiment.yaml

    # Run only one phase
    python run_experiments.py --config configs/full_experiment.yaml --phase 2

    # Skip training (use existing adapters) and go straight to evaluation
    python run_experiments.py --config configs/full_experiment.yaml --phase 2 3 4
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoTokenizer

# Add submodule paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "portability"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # Models
    source_model_id: str = "meta-llama/Meta-Llama-3-8B"
    target_model_id: Optional[str] = None      # for portability; None = skip

    # Training
    attributes: list = field(default_factory=lambda: ["sentiment", "topic"])
    datasets: dict = field(default_factory=lambda: {
        "sentiment": ["yelp_review_full", "imdb", "sst2"],
        "topic":     ["ag_news"],
    })
    seeds: list = field(default_factory=lambda: [8989, 79817, 794323])
    adapter_base_dir: str = "./trained_adapters"

    # Composition techniques to evaluate
    composition_modes: list = field(default_factory=lambda: [
        "sum", "average", "weight_average"
    ])

    # Portability
    n_cka_batches: int = 32
    max_layer_offset: int = 2
    cka_cache_dir: str = "./cka_cache"

    # Evaluation
    max_new_tokens: int = 100
    eval_batch_size: int = 16
    n_eval_prompts: int = 100           # prompts per attribute/dataset combo
    compute_slor: bool = True
    results_dir: str = "./results"

    # Device
    device: str = "cuda"
    use_4bit: bool = True


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    cfg = ExperimentConfig()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Phase 1: Training
# ---------------------------------------------------------------------------

def phase1_train(cfg: ExperimentConfig, dry_run: bool = False):
    """Train one RL LoRA per (attribute, dataset, seed) combination."""
    from train_grpo import TrainConfig, train

    logger.info("=" * 60)
    logger.info("PHASE 1: Training RL LoRA modules")
    logger.info("=" * 60)

    adapter_paths = {}   # (attribute, dataset, seed) -> path

    for attribute in cfg.attributes:
        datasets = cfg.datasets.get(attribute, [])
        for dataset_name in datasets:
            for seed in cfg.seeds:
                key = (attribute, dataset_name, seed)
                model_short = cfg.source_model_id.split("/")[-1]
                adapter_dir = os.path.join(
                    cfg.adapter_base_dir,
                    f"{model_short}_{attribute}_{dataset_name}_rl_seed{seed}"
                )

                if Path(adapter_dir, "adapter_model.safetensors").exists():
                    logger.info(f"  [SKIP] Already exists: {adapter_dir}")
                    adapter_paths[key] = adapter_dir
                    continue

                logger.info(f"  Training: attribute={attribute}, "
                            f"dataset={dataset_name}, seed={seed}")

                if dry_run:
                    logger.info("  [DRY RUN] Skipping actual training.")
                    adapter_paths[key] = adapter_dir
                    continue

                train_cfg = TrainConfig(
                    model_id=cfg.source_model_id,
                    attribute=attribute,
                    dataset_name=dataset_name,
                    seed=seed,
                    output_dir=cfg.adapter_base_dir,
                    use_4bit=cfg.use_4bit,
                )
                path = train(train_cfg)
                adapter_paths[key] = path

    return adapter_paths


# ---------------------------------------------------------------------------
# Phase 2: Composition evaluation
# ---------------------------------------------------------------------------

def phase2_composition(cfg: ExperimentConfig, adapter_paths: dict):
    """
    Evaluate single adapters and cross-attribute composition.

    With one dataset per attribute, there are no within-attribute combinations.
    The composition experiment is now purely cross-attribute:
        sentiment module + topic module → multi-attribute control

    Evaluation structure (mirrors Table 4 of the paper):
        - Single sentiment adapter  (Yelp)   — eval on sentiment prompts
        - Single topic adapter      (AG News) — eval on topic prompts
        - Composed (sentiment + topic):
            - Output Summing
            - Output Averaging
            - Weight Averaging
          — eval on both sentiment and topic prompts separately
    """
    from composition.compose import ComposedModel, compose_weight_average, load_single_adapter
    from evaluation.evaluate import Evaluator

    logger.info("=" * 60)
    logger.info("PHASE 2: Composition evaluation")
    logger.info("=" * 60)

    os.makedirs(cfg.results_dir, exist_ok=True)
    all_results = {}
    seed = cfg.seeds[0]

    tokenizer = AutoTokenizer.from_pretrained(cfg.source_model_id)

    # ---- Single adapters — one per attribute ----
    single_paths = {}
    for attribute in cfg.attributes:
        dataset_name = cfg.datasets.get(attribute, [""])[0]
        key = (attribute, dataset_name, seed)
        if key not in adapter_paths:
            logger.warning(f"Missing adapter for {key}, skipping.")
            continue

        logger.info(f"  Evaluating single adapter: {attribute} / {dataset_name}")
        evaluator = Evaluator(attribute, cfg.device, cfg.compute_slor)
        prompts, targets = _build_eval_prompts(attribute, n=cfg.n_eval_prompts)

        model = load_single_adapter(
            cfg.source_model_id, adapter_paths[key], use_4bit=cfg.use_4bit
        )
        results = evaluator.evaluate(
            model, tokenizer, prompts, targets,
            cfg.max_new_tokens, cfg.eval_batch_size
        )
        result_key = f"single_{attribute}"
        all_results[result_key] = results
        logger.info(f"    CE={results['CE']:.1f}, dist1={results['dist1']:.3f}")

        single_paths[attribute] = adapter_paths[key]
        del model
        torch.cuda.empty_cache()

    # ---- Cross-attribute composition: sentiment + topic ----
    if "sentiment" in single_paths and "topic" in single_paths:
        paths = {"sentiment": single_paths["sentiment"],
                 "topic":     single_paths["topic"]}

        for mode in cfg.composition_modes:
            logger.info(f"  Cross-attribute composition: {mode}")

            if mode == "weight_average":
                model = compose_weight_average(
                    cfg.source_model_id, paths, use_4bit=cfg.use_4bit
                )
            else:
                model = ComposedModel(
                    cfg.source_model_id, paths,
                    mode=mode, use_4bit=cfg.use_4bit
                )

            # Evaluate on both attributes separately — matches Table 4 structure
            for attribute in cfg.attributes:
                evaluator = Evaluator(attribute, cfg.device, compute_slor=False)
                prompts, targets = _build_eval_prompts(
                    attribute, n=cfg.n_eval_prompts
                )
                results = evaluator.evaluate(
                    model, tokenizer, prompts, targets,
                    cfg.max_new_tokens, cfg.eval_batch_size
                )
                result_key = f"composed_{mode}_{attribute}"
                all_results[result_key] = results
                logger.info(
                    f"    [{attribute}] CE={results['CE']:.1f}, "
                    f"dist1={results['dist1']:.3f}"
                )

            del model
            torch.cuda.empty_cache()
    else:
        logger.info("  Only one attribute available — skipping cross-attribute composition.")

    out_path = os.path.join(cfg.results_dir, "phase2_composition.json")
    _save_results(all_results, out_path)
    logger.info(f"Phase 2 results saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Phase 3: Portability evaluation
# ---------------------------------------------------------------------------

def phase3_portability(cfg: ExperimentConfig, adapter_paths: dict):
    """
    Port each RL-trained adapter to target_model_id and evaluate at 0 steps.
    The key new contribution: how much of the RL policy survives the transfer?
    """
    if cfg.target_model_id is None:
        logger.info("No target_model_id specified — skipping Phase 3.")
        return {}

    from portability.portable_peft import PortablePeftModel, detect_arch
    from portability.portable_peft import _load_adapter_weights
    from portability.cka_layer_mapping import cka_layer_mapping
    from evaluation.evaluate import evaluate_portability_delta
    from torch.utils.data import DataLoader
    from datasets import load_dataset

    logger.info("=" * 60)
    logger.info("PHASE 3: Portability evaluation (0 fine-tuning steps)")
    logger.info(f"  Source: {cfg.source_model_id}")
    logger.info(f"  Target: {cfg.target_model_id}")
    logger.info("=" * 60)

    os.makedirs(cfg.cka_cache_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Load both backbones (CPU to save VRAM during transform computation)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    logger.info("Loading source backbone...")
    src_model = AutoModelForCausalLM.from_pretrained(
        cfg.source_model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    tok_src = AutoTokenizer.from_pretrained(cfg.source_model_id)

    logger.info("Loading target backbone...")
    tgt_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model_id, torch_dtype=torch.float16, device_map="cpu"
    )
    tok_tgt = AutoTokenizer.from_pretrained(cfg.target_model_id)

    # Build calibration loader (used for CKA — in-domain data)
    calib_loader = _build_calib_loader(tok_src, n_samples=256, batch_size=8)

    # CKA matrix cache path
    src_short = cfg.source_model_id.split("/")[-1]
    tgt_short = cfg.target_model_id.split("/")[-1]
    cka_path = os.path.join(cfg.cka_cache_dir, f"cka_{src_short}_{tgt_short}.pt")

    all_results = {}

    for attribute in cfg.attributes:
        for dataset_name in cfg.datasets.get(attribute, []):
            seed = cfg.seeds[0]
            key = (attribute, dataset_name, seed)
            if key not in adapter_paths:
                continue

            lora_path = adapter_paths[key]
            logger.info(f"  Porting: {attribute}/{dataset_name} "
                        f"({src_short} -> {tgt_short})")

            # Port the adapter
            try:
                ported = PortablePeftModel.from_pretrained_ported(
                    new_model=tgt_model,
                    original_model=src_model,
                    lora_path=lora_path,
                    tokenizer_orig=tok_src,
                    tokenizer_new=tok_tgt,
                    calib_loader=calib_loader,
                    device=cfg.device,
                    n_cka_batches=cfg.n_cka_batches,
                    max_layer_offset=cfg.max_layer_offset,
                    cka_matrix_path=cka_path,   # reused across all adapters
                )
            except Exception as e:
                logger.error(f"Porting failed for {key}: {e}")
                continue

            # Save the ported adapter
            ported_dir = os.path.join(
                cfg.adapter_base_dir,
                f"ported_{tgt_short}_{attribute}_{dataset_name}_seed{seed}"
            )
            ported.save_ported_adapter(ported_dir)

            # Evaluate: original vs. ported
            prompts, targets = _build_eval_prompts(attribute, n=cfg.n_eval_prompts)

            # For original evaluation, load original adapter on source model
            from composition.compose import load_single_adapter
            orig_model = load_single_adapter(
                cfg.source_model_id, lora_path, use_4bit=cfg.use_4bit
            )

            delta = evaluate_portability_delta(
                original_model=orig_model,
                ported_model=ported,
                tokenizer_orig=tok_src,
                tokenizer_new=tok_tgt,
                prompts=prompts,
                target_labels=targets,
                attribute=attribute,
                device=cfg.device,
            )
            result_key = f"port_{attribute}_{dataset_name}_{tgt_short}"
            all_results[result_key] = delta
            logger.info(
                f"    CE_orig={delta['CE_original']:.1f}, "
                f"CE_ported={delta['CE_ported_0step']:.1f}, "
                f"CE_delta={delta['CE_delta']:.1f}, "
                f"retention={delta['CE_retention']:.3f}"
            )
            del orig_model, ported
            torch.cuda.empty_cache()

    out_path = os.path.join(cfg.results_dir, "phase3_portability.json")
    _save_results(all_results, out_path)
    logger.info(f"Phase 3 results saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Phase 4: Composition of ported modules
# ---------------------------------------------------------------------------

def phase4_composed_ported(cfg: ExperimentConfig):
    """
    After porting, evaluate whether composed ported modules still work.
    This answers: can you port modules independently and then compose them?
    """
    if cfg.target_model_id is None:
        logger.info("No target_model_id — skipping Phase 4.")
        return {}

    from composition.compose import ComposedModel, compose_weight_average
    from evaluation.evaluate import Evaluator

    logger.info("=" * 60)
    logger.info("PHASE 4: Composition of ported modules")
    logger.info("=" * 60)

    tgt_short = cfg.target_model_id.split("/")[-1]
    seed = cfg.seeds[0]
    all_results = {}

    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_id)

    # Build adapter paths for ported modules
    ported_paths = {}
    for attribute in cfg.attributes:
        for dataset_name in cfg.datasets.get(attribute, []):
            ported_dir = os.path.join(
                cfg.adapter_base_dir,
                f"ported_{tgt_short}_{attribute}_{dataset_name}_seed{seed}"
            )
            if Path(ported_dir).exists():
                ported_paths[(attribute, dataset_name)] = ported_dir
            else:
                logger.warning(f"Ported adapter not found: {ported_dir}")

    # Cross-attribute composition (sentiment + topic)
    if "sentiment" in cfg.attributes and "topic" in cfg.attributes:
        s_datasets = cfg.datasets.get("sentiment", [])
        t_datasets = cfg.datasets.get("topic", [])

        for s_ds in s_datasets:
            for t_ds in t_datasets:
                s_key = ("sentiment", s_ds)
                t_key = ("topic", t_ds)
                if s_key not in ported_paths or t_key not in ported_paths:
                    continue

                paths = {
                    f"sentiment_{s_ds}": ported_paths[s_key],
                    f"topic_{t_ds}":     ported_paths[t_key],
                }
                combo_str = f"sentiment_{s_ds}+topic_{t_ds}"
                logger.info(f"  Composed ported: {combo_str}")

                for mode in cfg.composition_modes:
                    if mode == "weight_average":
                        model = compose_weight_average(
                            cfg.target_model_id, paths, use_4bit=cfg.use_4bit
                        )
                    else:
                        model = ComposedModel(
                            cfg.target_model_id, paths,
                            mode=mode, use_4bit=cfg.use_4bit
                        )

                    # Evaluate multi-attribute CE
                    from evaluation.evaluate import ControlEffectivenessScorer
                    s_prompts, s_targets = _build_eval_prompts("sentiment",
                                                                n=cfg.n_eval_prompts // 2)
                    t_prompts, t_targets = _build_eval_prompts("topic",
                                                                n=cfg.n_eval_prompts // 2)

                    evaluator_s = Evaluator("sentiment", cfg.device, compute_slor=False)
                    evaluator_t = Evaluator("topic", cfg.device, compute_slor=False)

                    r_s = evaluator_s.evaluate(model, tokenizer, s_prompts, s_targets,
                                               cfg.max_new_tokens, cfg.eval_batch_size)
                    r_t = evaluator_t.evaluate(model, tokenizer, t_prompts, t_targets,
                                               cfg.max_new_tokens, cfg.eval_batch_size)

                    result_key = f"{mode}_ported_{combo_str}"
                    all_results[result_key] = {
                        "CE_sentiment": r_s["CE"],
                        "CE_topic":     r_t["CE"],
                        "dist1": (r_s["dist1"] + r_t["dist1"]) / 2,
                    }
                    logger.info(f"    CE_sentiment={r_s['CE']:.1f}, CE_topic={r_t['CE']:.1f}")
                    del model
                    torch.cuda.empty_cache()

    out_path = os.path.join(cfg.results_dir, "phase4_composed_ported.json")
    _save_results(all_results, out_path)
    logger.info(f"Phase 4 results saved to {out_path}")
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_eval_prompts(attribute: str, n: int = 100) -> tuple[list, list]:
    """Build evaluation prompts matching the paper's format."""
    import random
    if attribute == "sentiment":
        labels = ["Positive", "Negative"]
        starters = [
            "The food was", "I really", "This place", "The service",
            "My experience", "Everything about", "I would", "The staff",
            "", "After visiting",
        ]
    else:  # topic
        labels = ["World", "Sports", "Business", "Science/Technology"]
        starters = [
            "The government", "The team", "The company", "Researchers",
            "Scientists", "The president", "The market", "Players",
            "", "A new study",
        ]

    prompts, targets = [], []
    for i in range(n):
        label = labels[i % len(labels)]
        starter = random.choice(starters)
        attr_tag = "SENTIMENT" if attribute == "sentiment" else "TOPIC"
        prompt = f"[{attr_tag}] {label} [\\{attr_tag}] [ANS] {starter}".strip()
        prompts.append(prompt)
        targets.append(label)

    return prompts, targets


def _build_calib_loader(tokenizer, n_samples: int = 256, batch_size: int = 8):
    """Small calibration DataLoader for CKA computation."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    texts = load_dataset("wikitext", "wikitext-2-raw-v1",
                         split="train")["text"][:n_samples]
    texts = [t for t in texts if len(t.strip()) > 20][:n_samples]

    def collate(batch):
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256)
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]}

    return DataLoader(texts, batch_size=batch_size,
                      collate_fn=collate, shuffle=False)


def _save_results(results: dict, path: str):
    """Save results to JSON, stripping non-serialisable fields (generated texts)."""
    serialisable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serialisable[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, list) or kk != "generated_texts"
            }
        else:
            serialisable[k] = v
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--phase", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which phases to run (1=train, 2=compose, 3=port, 4=port+compose)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip actual training, use existing adapters")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    logger.info(f"Running phases: {args.phase}")

    adapter_paths = {}

    if 1 in args.phase:
        adapter_paths = phase1_train(cfg, dry_run=args.dry_run)
    else:
        # Reconstruct adapter paths from disk
        for attr in cfg.attributes:
            for ds in cfg.datasets.get(attr, []):
                for seed in cfg.seeds:
                    model_short = cfg.source_model_id.split("/")[-1]
                    d = os.path.join(
                        cfg.adapter_base_dir,
                        f"{model_short}_{attr}_{ds}_rl_seed{seed}"
                    )
                    if Path(d).exists():
                        adapter_paths[(attr, ds, seed)] = d
        logger.info(f"Found {len(adapter_paths)} existing adapters.")

    if 2 in args.phase:
        phase2_composition(cfg, adapter_paths)

    if 3 in args.phase:
        phase3_portability(cfg, adapter_paths)

    if 4 in args.phase:
        phase4_composed_ported(cfg)

    logger.info("All requested phases complete.")
