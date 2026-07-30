"""
training/train_post_porting_sft.py

Post-porting light fine-tuning (LFT):
  1. Port a trained LoRA module from src_model to tgt_model using the
     algebraic transform from run_portability.py (0 gradient steps).
  2. Fine-tune the ported module on the target backbone for a small number
     of steps using the standard SFT objective (completion-only CE loss).
  3. Save the best checkpoint (selected by eval CE on the validation set).
  4. Generate outputs on the given test datasets using the trained model
     (no reload - the model is already in memory after training).

This directly mirrors the "Light Fine-Tuning" (LFT) condition in LoRASuite,
extended to cross-family portability.  The key hyperparameter is lft_max_steps
in the config (default 200 - the same checkpoint used in the paper's best runs).

Usage
-----
    python training/train_post_porting_sft.py \\
        --config_path  configs/lft_config.yaml \\
        --src_model    llama31 \\
        --tgt_model    llama32 \\
        --module       models/sft_llama3.1_8b_sentiment_seed7097/checkpoint-200 \\
        --attribute    sentiment \\
        --seed         7097 \\
        --output_dir   models/lft_llama31_to_llama32_sentiment_seed7097 \\
        --test_datasets data/pplm_prompts.csv data/sts_benchmark_test_subset.csv data/sts_benchmark_processed.csv \\
        --output_json  results/lft_llama31_to_llama32_sentiment_seed7097.json

The config YAML is the same format as the SFT training config.  The only
fields read here are: data, training, model, max_seq_length, quantization
(optional), wandb_project.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# The porting logic lives in run_portability.py - import it directly so we
# don't duplicate the algebraic transform.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "portability"))
from run_portability import (
    get_ported_module, load_eval_prompts,
    MODEL_IDS, TRANSFER_CONDITION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data preparation - identical to train_sft_model.py
# ---------------------------------------------------------------------------

def sample_to_prompt_completion(sample: dict, attribute: str = "sentiment") -> dict:
    """
    Format a training sample into the CTG prompt/completion format:
        [SENTIMENT] Positive [\\SENTIMENT] [ANS] {input} {output} [\\ANS]
    """
    prompt = (
        f"[{attribute.upper()}] {sample['control']} "
        f"[\\{attribute.upper()}] [ANS] {sample['input']}"
    ).strip()
    completion = f" {sample['output']} [\\ANS]"
    return {"prompt": prompt, "completion": completion}


# ---------------------------------------------------------------------------
# Generation - runs on an already-loaded model, no reload needed
# ---------------------------------------------------------------------------

def generate_with_model(
    model,
    tokenizer,
    dataset_names: list[str],
    attribute: str,
    max_new_tokens: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Generate completions for each test dataset using an already-loaded model.

    The tokenizer is temporarily switched to left-padding for generation
    (required for correct batched autoregressive decoding), then restored.

    Returns a dict keyed by dataset path:
        {
            dataset_path: {
                "generated_texts": [{prompt, raw_completion, completion, label}, ...],
                "start_time": "...",
                "end_time":   "...",
            },
            ...
        }
    """
    # Switch to left-padding for generation - right-padding causes the model
    # to attend to padding tokens to the right of real content.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    model.eval()
    results = {}

    for dataset_name in dataset_names:
        start_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"  Generating for: {dataset_name}")

        records = load_eval_prompts(dataset_name, attribute)
        prompts = [r["prompt"] for r in records]
        labels  = [r["label"]  for r in records]

        generated_texts = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            batch_labels  = labels[i: i + batch_size]

            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens    = max_new_tokens,
                    do_sample         = False,
                    pad_token_id      = tokenizer.pad_token_id,
                    repetition_penalty= 1.3,
                    no_repeat_ngram_size = 4,
                )

            for ids, prompt, label in zip(out_ids, batch_prompts, batch_labels):
                raw   = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
                clean = raw.replace("[\\ANS]", "").replace("[ANS]", "").strip()
                generated_texts.append({
                    "prompt":         prompt,
                    "raw_completion": raw,
                    "completion":     clean,
                    "label":          label,
                })

            logger.info(f"    {min(i + batch_size, len(prompts))}/{len(prompts)} done")

        end_time = datetime.now(timezone.utc).isoformat()
        results[dataset_name] = {
            "generated_texts": generated_texts,
            "start_time":      start_time,
            "end_time":        end_time,
        }
        logger.info(f"    → {len(generated_texts)} completions generated")

    # Restore original padding side for any subsequent use
    tokenizer.padding_side = orig_padding_side
    model.train()   # restore training mode

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-porting light fine-tuning + generation."
    )
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the YAML training config.")
    parser.add_argument("--src_model", type=str, required=True,
                        choices=list(MODEL_IDS.keys()),
                        help="Source model key (e.g. llama31).")
    parser.add_argument("--tgt_model", type=str, required=True,
                        choices=list(MODEL_IDS.keys()),
                        help="Target model key (e.g. llama32).")
    parser.add_argument("--module", type=str, required=True,
                        help="Path to the trained SFT LoRA checkpoint to port.")
    parser.add_argument("--attribute", type=str, default="sentiment",
                        choices=["sentiment", "topic"],
                        help="Control attribute.")
    parser.add_argument("--seed", type=int, default=7097,
                        help="Random seed (should match the source module seed).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save the post-porting LFT checkpoint.")
    parser.add_argument("--test_datasets", type=str, nargs="+", required=True,
                        help="List of test CSV paths to generate on after training.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to write the generation output JSON.")
    parser.add_argument("--cka_dir", type=str, default="./cka_cache",
                        help="Directory containing pre-computed CKA matrices.")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Max tokens to generate per prompt.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Generation batch size.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for porting computation.")
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    run_name = (
        f"lft_{args.src_model}_to_{args.tgt_model}"
        f"_{args.attribute}_seed{args.seed}"
    )
    os.environ["WANDB_PROJECT"] = config.get("wandb_project", "portability_lft")
    wandb.init(
        project=config.get("wandb_project", "portability_lft"),
        name=run_name,
        tags=[args.src_model, args.tgt_model, args.attribute, f"seed{args.seed}"],
    )
    wandb.config.update({
        "src_model":     args.src_model,
        "tgt_model":     args.tgt_model,
        "module":        args.module,
        "attribute":     args.attribute,
        "seed":          args.seed,
        "test_datasets": args.test_datasets,
    })

    # ------------------------------------------------------------------
    # Step 1: Port the module (0 gradient steps)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Step 1: Porting {args.src_model} → {args.tgt_model}")
    logger.info("=" * 60)

    ported_model, tok_tgt, start_port, end_port = get_ported_module(
        module_path   = args.module,
        tgt_model_key = args.tgt_model,
        src_model_key = args.src_model,
        cka_dir       = args.cka_dir,
        device        = args.device,
    )

    # Save ported weights to temp dir so SFTTrainer can reload with
    # quantization (get_ported_module uses float32 for numerical precision
    # during the algebraic transform; SFT needs bfloat16 + 4-bit).
    tmp_dir = tempfile.mkdtemp(prefix="ported_module_")
    logger.info(f"Saving ported module to temp dir: {tmp_dir}")
    ported_model.save_pretrained(tmp_dir)

    del ported_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 2: Reload with quantization and fine-tune
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: Light fine-tuning on target backbone")
    logger.info("=" * 60)

    # Tokenizer - right-padding for SFT training
    tok_tgt.padding_side = "right"
    if hasattr(tok_tgt, "enable_thinking"):
        tok_tgt.enable_thinking = False

    pad_token = config.get("model", {}).get("pad_token", None)
    if pad_token and pad_token in tok_tgt.get_vocab():
        tok_tgt.pad_token    = pad_token
        tok_tgt.pad_token_id = tok_tgt.convert_tokens_to_ids(pad_token)
    elif tok_tgt.pad_token is None:
        tok_tgt.pad_token    = tok_tgt.eos_token
        tok_tgt.pad_token_id = tok_tgt.eos_token_id
    logger.info(f"Pad token: '{tok_tgt.pad_token}' (id={tok_tgt.pad_token_id})")

    # Datasets
    train_dataset = load_dataset("csv", data_files=config["data"]["train_file"])["train"]
    eval_dataset  = load_dataset("csv", data_files=config["data"]["val_file"])["train"]
    train_dataset = train_dataset.map(
        sample_to_prompt_completion, fn_kwargs={"attribute": args.attribute}
    )
    eval_dataset = eval_dataset.map(
        sample_to_prompt_completion, fn_kwargs={"attribute": args.attribute}
    )
    logger.info(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    # Quantization
    bnb_config = None
    if "quantization" in config:
        compute_dtype = getattr(torch, config["quantization"]["bnb_4bit_compute_dtype"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit              = config["quantization"]["use_4bit"],
            bnb_4bit_quant_type       = config["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype    = compute_dtype,
            bnb_4bit_use_double_quant = config["quantization"]["use_nested_quant"],
        )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[args.tgt_model],
        torch_dtype         = torch.bfloat16,
        quantization_config = bnb_config,
    )
    if bnb_config is not None and bnb_config.load_in_4bit:
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base)

    model = PeftModel.from_pretrained(base, tmp_dir, is_trainable=True)
    model.print_trainable_parameters()

    lft_steps = config["training"].get("lft_max_steps", 200)
    eval_steps = config["training"].get("eval_steps", 50)
    logger.info(f"LFT steps: {lft_steps}  |  Eval every: {eval_steps} steps")

    train_args = SFTConfig(
        output_dir                  = args.output_dir,
        seed                        = args.seed,
        data_seed                   = args.seed,
        max_length                  = config["max_seq_length"],
        per_device_train_batch_size = config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size  = config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
        learning_rate               = float(config["training"]["learning_rate"]),
        weight_decay                = config["training"]["weight_decay"],
        optim                       = config["training"]["optim"],
        fp16                        = config["training"].get("fp16", False),
        bf16                        = config["training"].get("bf16", True),
        max_grad_norm               = config["training"]["max_grad_norm"],
        warmup_ratio                = config["training"].get("warmup_ratio", 0.1),
        lr_scheduler_type           = config["training"].get("lr_scheduler_type", "cosine"),
        max_steps                   = lft_steps,
        save_strategy               = "steps",
        save_steps                  = 200,
        save_total_limit            = 3,
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_strategy            = "steps",
        logging_steps               = config["training"].get("logging_steps", 10),
        report_to                   = "wandb",
        run_name                    = run_name,
        completion_only_loss        = True,
    )

    trainer = SFTTrainer(
        model            = model,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        processing_class = tok_tgt,
        args             = train_args,
    )

    logger.info("Starting LFT training...")
    trainer.train()

    # Save best checkpoint and tokenizer
    logger.info(f"Saving best checkpoint to {args.output_dir}...")
    #trainer.save_model(args.output_dir)
    #tok_tgt.save_pretrained(args.output_dir)

    # ------------------------------------------------------------------
    # Step 3: Generate on test datasets - model is already in memory
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Generating on test datasets")
    logger.info(f"  Datasets: {args.test_datasets}")
    logger.info("=" * 60)

    condition = TRANSFER_CONDITION.get(
        (args.src_model, args.tgt_model), "unknown"
    )

    # trainer.model holds the best checkpoint after load_best_model_at_end
    lft_model = trainer.model

    test_sets = generate_with_model(
        model         = lft_model,
        tokenizer     = tok_tgt,
        dataset_names = args.test_datasets,
        attribute     = args.attribute,
        max_new_tokens= args.max_new_tokens,
        batch_size    = 1,
        device        = args.device,
    )

    # ------------------------------------------------------------------
    # Save generation output - same JSON schema as run_portability.py
    # ------------------------------------------------------------------
    lft_steps_done = config["training"].get("lft_max_steps", 200)
    output = {
        "start_time_port": start_port,
        "end_time_port":   end_port,
        "model":           MODEL_IDS[args.tgt_model],
        "module":          args.output_dir,   # the LFT checkpoint
        "src_model":       args.src_model,
        "tgt_model":       args.tgt_model,
        "condition":       condition,
        "attribute":       args.attribute,
        "lft_steps":       lft_steps_done,
        "porting":         "algebraic_transform + LFT",
        "test_sets":       test_sets,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Generation saved to: {args.output_json}")

    # Save metadata alongside the checkpoint
    metadata = {
        "src_model":    args.src_model,
        "tgt_model":    args.tgt_model,
        "src_module":   args.module,
        "attribute":    args.attribute,
        "seed":         args.seed,
        "lft_steps":    lft_steps_done,
        "porting":      "algebraic_transform + LFT",
        "output_dir":   args.output_dir,
        "output_json":  args.output_json,
    }
    with open(os.path.join(args.output_dir, "lft_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
