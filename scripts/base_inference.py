
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import torch

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("base_inference")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_IDS = {
    "llama32": "meta-llama/Llama-3.2-3B",
    "llama31": "meta-llama/Llama-3.1-8B",
    "qwen34b": "Qwen/Qwen3-4B",
    "qwen38b": "Qwen/Qwen3-8B",
}


# ---------------------------------------------------------------------------
# Dataset loaders — returns list of {"prompt": str, "attribute": str, "label": str}
# ---------------------------------------------------------------------------


def load_eval_prompts(
    dataset_name: str,
    attribute: str,
) -> list[dict]:
    """
    Load evaluation prompts from a dataset.
    Returns a list of {"prompt": str, "attribute": str, "label": str} dicts
    where `label` is the target label for CE scoring.

    For out-of-domain datasets (pplm, sts) labels are assigned
    by cycling through the attribute's label set, matching the
    paper's evaluation setup.
    """
    from datasets import load_dataset

    if attribute == "sentiment":
        labels    = ["Positive", "Negative"]
        tag       = "SENTIMENT"
    elif attribute == "topic":
        labels    = ["World", "Sports", "Business", "Science/Technology"]
        tag       = "TOPIC"
    else:
        raise ValueError(f"Unknown attribute: {attribute}. Use 'sentiment' or 'topic'.")

    records = []

    test_data = load_dataset("csv", data_files=dataset_name)['train']
    # iterate over test_data and create prompts
    for ex in test_data:
        for val in labels:
            prompt = f"[{tag}] {val} [\{tag}] [ANS] {ex['input']}".strip()
            records.append({"prompt": prompt, "attribute": attribute, "label": val})

    logger.info(f"Loaded {len(records)} prompts from {dataset_name} ({attribute})")
    return records


def step_generate(
    tgt_model_key: str,
    dataset_names: List[str],
    attribute: str,
    output_path: str,
    max_new_tokens: int = 100,
    batch_size: int = 16,
    device: str = "cuda",
):
    """
    Load a module (porting it if src_model_key != tgt_model_key),
    generate texts for the given dataset, and save to output_path.

    If src_model_key is None or equal to tgt_model_key: identity run (no porting).
    Otherwise: port the module algebraically from src to tgt before inference.

    Output JSON schema:
    {
      "start_time":  "...",
      "end_time":    "...",
      "model":       "meta-llama/Llama-3.2-3B",
      "module":      "./models/sft_llama3.2_3b_sentiment_seed7097/checkpoint-200",
      "src_model":   "llama32",
      "tgt_model":   "qwen34b",
      "condition":   "cross_family_comparable_size",  (or "identity")
      "dataset":     "yelp",
      "attribute":   "sentiment",
      "test_sets": {
        "test_data1": {
            "generated_texts": [
                {
                "prompt":         "[SENTIMENT] Positive [\\SENTIMENT] [ANS] The food was",
                "raw_completion": "absolutely fantastic! Best meal ever. [\\ANS]",
                "completion":     "absolutely fantastic! Best meal ever.",
                "label":      "Positive"
                }, ...
            ],
            "start_time": "...",
            "end_time":   "..."
        }
      }
    }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"{'='*60}")

    tok_tgt = AutoTokenizer.from_pretrained(MODEL_IDS[tgt_model_key])
    if hasattr(tok_tgt, "enable_thinking"):
        tok_tgt.enable_thinking = False

    pad_token = "<|endoftext|>" if "Qwen" in MODEL_IDS[tgt_model_key] else "<|finetune_right_pad_id|>"
 
    if pad_token in tok_tgt.get_vocab():
        tok_tgt.pad_token = pad_token
        tok_tgt.pad_token_id = tok_tgt.convert_tokens_to_ids(pad_token)
    elif pad_token:
        tok_tgt.add_special_tokens({'pad_token': pad_token})
    elif tok_tgt.pad_token is None:
        tok_tgt.pad_token = tok_tgt.eos_token
        tok_tgt.pad_token_id = tok_tgt.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS[tgt_model_key],
        torch_dtype=torch.bfloat16
    ).to(device)

    output = {
        "start_time_port":     None,
        "end_time_port":       None,
        "model":          MODEL_IDS[tgt_model_key],
        "src_model":      None,
        "tgt_model":      tgt_model_key,
        "condition":      None,
        "attribute":      attribute,
        "test_sets": {},
    }
    for dataset_name in dataset_names:
        start_time = datetime.now(timezone.utc).isoformat()
        logger.info(f"  Dataset: {dataset_name}  Attribute: {attribute}")

        # ---- Load prompts ----
        records = load_eval_prompts(dataset_name, attribute)
        prompts  = [r["prompt"]    for r in records]
        attrs    = [r["label"] for r in records]

        # ---- Generate ----
        logger.info(f"Generating {len(prompts)} completions...")
        generated_texts = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            batch_attrs   = attrs[i: i + batch_size]

            enc = tok_tgt(
                batch_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=1024,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok_tgt.eos_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=4
                )

            for ids, prompt, attr in zip(out_ids, batch_prompts, batch_attrs):
                raw = tok_tgt.decode(ids[prompt_len:], skip_special_tokens=True)
                clean = raw.replace("[\ANS]", "").replace("[ANS]", "").strip()
                generated_texts.append({
                    "prompt":         prompt,
                    "raw_completion": raw,
                    "completion":     clean,
                    "label":      attr,
                })

            logger.info(f"  {min(i + batch_size, len(prompts))}/{len(prompts)} done")

        end_time = datetime.now(timezone.utc).isoformat()

        # ---- Save ----
        output["test_sets"][dataset_name] = {
            "generated_texts": generated_texts,
            "start_time": start_time,
            "end_time": end_time
        }

        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(generated_texts)} texts → {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )


    # --- generate ---
    parser.add_argument("--tgt_model",  type=str, default=None,
                        choices=list(MODEL_IDS.keys()),
                        help="Model to run inference on (after porting if needed).")
    parser.add_argument("--dataset", type=str, nargs="+", required=True,
                        help="List of test CSV paths to generate on after training.")
    parser.add_argument("--attribute",  type=str, default="sentiment",
                        choices=["sentiment", "topic"],
                        help="Control attribute.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output",     type=str, default=None,
                        help="Output JSON path for --step generate or score.")

    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--seed",     type=int, default=7097)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Auto-build output filename if not specified
    output = args.output
    if output is None:
        logger.error("No --output specified")

    step_generate(
        tgt_model_key=args.tgt_model,
        dataset_names=args.dataset,
        attribute=args.attribute,
        output_path=output,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )
