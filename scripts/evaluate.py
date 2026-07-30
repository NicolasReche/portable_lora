"""
evaluation/evaluate.py

Evaluates a model (single adapter, composed adapters, or ported adapter)
using the same three metrics as the paper:
    - Diversity:  Distinct-1, Distinct-2, Distinct-3
    - Fluency:    SLOR (using GPT-2-XL and BLOOM-1B7)
    - Control Effectiveness (CE): classifier ensemble vote

Usage:
    from evaluation.evaluate import Evaluator

    evaluator = Evaluator(attribute="sentiment")
    results = evaluator.evaluate(
        model=my_model,
        tokenizer=tok,
        prompts=["[SENTIMENT] Positive [\\SENTIMENT] [ANS] The food"],
        target_labels=["Positive"],
    )
    print(results)  # {"CE": 0.88, "dist1": 0.12, "dist2": 0.34, "SLOR": 9.5}
"""

import math
import logging
from collections import Counter
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Control Effectiveness
# ---------------------------------------------------------------------------

class ControlEffectivenessScorer:
    """
    Ensemble of classifiers voting on whether generated text has the
    target attribute. Matches the CE metric from the paper exactly.
    """

    def __init__(self, attribute: str, device: str = "cuda"):
        self.attribute = attribute
        dev = 0 if device == "cuda" and torch.cuda.is_available() else -1

        if attribute == "sentiment":
            self.classifiers = [
                pipeline("text-classification",
                         model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                         device=dev),
                pipeline("text-classification",
                         model="michelecafagna26/t5-base-finetuned-sst2-sentiment",
                         device=dev),
            ]
            self.label_map = {
                "POSITIVE": "Positive", "NEGATIVE": "Negative",
                "LABEL_1": "Positive", "LABEL_0": "Negative",
            }

        elif attribute == "topic":
            self.classifiers = [
                pipeline("text-classification",
                         model="textattack/distilbert-base-uncased-ag-news",
                         device=dev),
                pipeline("text-classification",
                         model="fabriceyhc/bert-base-uncased-ag_news",
                         device=dev),
            ]
            self.label_map = {
                "LABEL_0": "World", "LABEL_1": "Sports",
                "LABEL_2": "Business", "LABEL_3": "Science/Technology",
            }
        else:
            raise ValueError(f"Unknown attribute: {attribute}")

    def score(self, texts: list[str], target_labels: list[str]) -> float:
        """Returns mean CE across all texts (0-100 scale, matching the paper)."""
        per_clf_scores = []
        for clf in self.classifiers:
            correct = 0
            for text, target in zip(texts, target_labels):
                try:
                    result = clf(text[:512], truncation=True)[0]
                    predicted = self.label_map.get(result["label"].upper(), "")
                    if predicted == target:
                        correct += 1
                except Exception:
                    pass
            per_clf_scores.append(100 * correct / len(texts) if texts else 0.0)
        return sum(per_clf_scores) / len(per_clf_scores)

    def score_multi(
        self,
        texts: list[str],
        target_sentiment: list[str],
        target_topic: list[str],
        sentiment_scorer: "ControlEffectivenessScorer",
        topic_scorer: "ControlEffectivenessScorer",
    ) -> float:
        """
        Multi-attribute CE: both sentiment AND topic must be correct.
        Uses majority voting across classifiers for each attribute,
        then requires both to match.
        """
        correct = 0
        for text, s_target, t_target in zip(texts, target_sentiment, target_topic):
            text_trunc = text[:512]
            # Majority vote for sentiment
            s_votes = []
            for clf in sentiment_scorer.classifiers:
                try:
                    r = clf(text_trunc, truncation=True)[0]
                    s_votes.append(
                        sentiment_scorer.label_map.get(r["label"].upper(), "")
                    )
                except Exception:
                    pass
            s_pred = Counter(s_votes).most_common(1)[0][0] if s_votes else ""

            # Majority vote for topic
            t_votes = []
            for clf in topic_scorer.classifiers:
                try:
                    r = clf(text_trunc, truncation=True)[0]
                    t_votes.append(
                        topic_scorer.label_map.get(r["label"].upper(), "")
                    )
                except Exception:
                    pass
            t_pred = Counter(t_votes).most_common(1)[0][0] if t_votes else ""

            if s_pred == s_target and t_pred == t_target:
                correct += 1

        return 100 * correct / len(texts) if texts else 0.0


# ---------------------------------------------------------------------------
# Diversity: Distinct-n
# ---------------------------------------------------------------------------

def distinct_n(texts: list[str], n: int) -> float:
    """
    Distinct-n: proportion of unique n-grams across all texts.
    Matches the item-level mean implementation from the paper.
    """
    item_scores = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n:
            item_scores.append(0.0)
            continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            item_scores.append(0.0)
        else:
            item_scores.append(len(set(ngrams)) / len(ngrams))
    return sum(item_scores) / len(item_scores) if item_scores else 0.0


# ---------------------------------------------------------------------------
# Fluency: SLOR
# ---------------------------------------------------------------------------

class SLORScorer:
    """
    Syntactic Log-Odds Ratio fluency metric.
    SLOR(s) = (log P(s) - log P_unigram(s)) / len(s)

    Uses GPT-2-XL and BLOOM-1B7 as in the paper, averages both.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device

        logger.info("Loading SLOR models (GPT-2-XL and BLOOM-1B7)...")
        self.models = []
        for model_id in ["gpt2-xl", "bigscience/bloom-1b7"]:
            tok = AutoTokenizer.from_pretrained(model_id)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
            mdl.eval()
            self.models.append((tok, mdl))

    @torch.no_grad()
    def _log_prob(self, tokenizer, model, text: str) -> tuple[float, int]:
        """Returns (sentence log-prob, token count)."""
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(self.device)
        input_ids = enc["input_ids"]
        n_tokens = input_ids.shape[1]
        if n_tokens < 2:
            return 0.0, 0

        out = model(input_ids, labels=input_ids)
        # out.loss is mean NLL per token; total log-prob = -loss * n_tokens
        log_prob = -out.loss.item() * n_tokens
        return log_prob, n_tokens

    @torch.no_grad()
    def _unigram_log_prob(self, tokenizer, model, text: str) -> float:
        """Sum of log P(token) for each token independently."""
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=512).to(self.device)
        input_ids = enc["input_ids"][0]
        total = 0.0
        for tok_id in input_ids:
            single = tok_id.unsqueeze(0).unsqueeze(0)
            out = model(single, labels=single)
            total += -out.loss.item()
        return total

    def score(self, texts: list[str]) -> float:
        """Returns mean SLOR across texts, averaged over both LMs."""
        all_slors = []
        for text in texts:
            if not text.strip():
                continue
            per_model_slors = []
            for tokenizer, model in self.models:
                log_p, n = self._log_prob(tokenizer, model, text)
                if n < 2:
                    continue
                log_p_uni = self._unigram_log_prob(tokenizer, model, text)
                slor = (log_p - log_p_uni) / n
                per_model_slors.append(slor)
            if per_model_slors:
                all_slors.append(sum(per_model_slors) / len(per_model_slors))
        return sum(all_slors) / len(all_slors) if all_slors else 0.0


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Runs all three evaluation metrics on a set of generated texts.
    Designed to be model-agnostic: pass any model with a .generate() method.
    """

    def __init__(
        self,
        attribute: str,
        device: str = "cuda",
        compute_slor: bool = True,   # set False during debugging to save time
    ):
        self.attribute = attribute
        self.device = device
        self.ce_scorer = ControlEffectivenessScorer(attribute, device)
        self.slor_scorer = SLORScorer(device) if compute_slor else None

    def generate_texts(
        self,
        model,
        tokenizer,
        prompts: list[str],
        max_new_tokens: int = 100,
        batch_size: int = 16,
    ) -> list[str]:
        """Generate one completion per prompt."""
        all_texts = []
        tokenizer.padding_side = "left"

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            enc = tokenizer(
                batch_prompts, return_tensors="pt",
                padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the generated part (strip the prompt)
            prompt_len = enc["input_ids"].shape[1]
            for ids in out_ids:
                generated = tokenizer.decode(
                    ids[prompt_len:], skip_special_tokens=True
                )
                # Strip answer tags
                generated = generated.replace("[\\ANS]", "").replace("[ANS]", "").strip()
                all_texts.append(generated)

        return all_texts

    def evaluate(
        self,
        model,
        tokenizer,
        prompts: list[str],
        target_labels: list[str],
        max_new_tokens: int = 100,
        batch_size: int = 16,
    ) -> dict:
        """
        Full evaluation pipeline. Returns a dict matching the paper's columns:
            CE, dist1, dist2, dist3, SLOR
        """
        logger.info(f"Generating {len(prompts)} texts...")
        texts = self.generate_texts(model, tokenizer, prompts,
                                    max_new_tokens, batch_size)

        results = {}

        # Control Effectiveness
        logger.info("Computing CE...")
        results["CE"] = self.ce_scorer.score(texts, target_labels)

        # Diversity
        results["dist1"] = distinct_n(texts, 1)
        results["dist2"] = distinct_n(texts, 2)
        results["dist3"] = distinct_n(texts, 3)

        # Fluency
        if self.slor_scorer is not None:
            logger.info("Computing SLOR (slow)...")
            results["SLOR"] = self.slor_scorer.score(texts)
        else:
            results["SLOR"] = None

        results["generated_texts"] = texts
        return results


# ---------------------------------------------------------------------------
# Signal retention analysis (for portability experiments)
# ---------------------------------------------------------------------------

def evaluate_portability_delta(
    original_model,      # PeftModel with RL-trained adapter
    ported_model,        # PortablePeftModel after 0-step porting
    tokenizer_orig,
    tokenizer_new,
    prompts: list[str],
    target_labels: list[str],
    attribute: str,
    device: str = "cuda",
) -> dict:
    """
    Compare CE of the original RL adapter vs the ported adapter.
    This is the key number for your portability experiments:
        - CE_original: baseline on original model
        - CE_ported_0step: after algebraic transform, 0 fine-tuning
        - CE_delta: the gap (tells you how much policy signal survives)
    """
    evaluator_orig = Evaluator(attribute, device, compute_slor=False)
    evaluator_new  = Evaluator(attribute, device, compute_slor=False)

    logger.info("Evaluating original RL adapter...")
    orig_results = evaluator_orig.evaluate(
        original_model, tokenizer_orig, prompts, target_labels
    )

    logger.info("Evaluating ported adapter (0 steps)...")
    ported_results = evaluator_new.evaluate(
        ported_model, tokenizer_new, prompts, target_labels
    )

    return {
        "CE_original":    orig_results["CE"],
        "CE_ported_0step": ported_results["CE"],
        "CE_delta":       ported_results["CE"] - orig_results["CE"],
        "CE_retention":   ported_results["CE"] / (orig_results["CE"] + 1e-6),
        "original_texts": orig_results["generated_texts"],
        "ported_texts":   ported_results["generated_texts"],
    }
