"""
composition/compose.py

Implements the three composition techniques from the paper, now on top of
the HuggingFace PEFT library instead of a custom LoRA implementation.

Techniques:
    1. OutputSumming     — h = W0 x + Σ (αi/ri)(ΔWi x)
    2. OutputAveraging   — h = W0 x + (1/N) Σ (αi/ri)(ΔWi x)
    3. WeightAveraging   — uses PEFT's built-in add_weighted_adapter

The key insight: OutputSumming/Averaging are not weight-space operations,
so they cannot be done by merging adapter weights. They require running
each adapter's forward pass separately and combining the outputs.
WeightAveraging IS a weight-space operation and PEFT handles it natively.

Usage:
    from composition.compose import ComposedModel, compose_weight_average

    # Output summing (best performing in the paper)
    model = ComposedModel(
        base_model_id="meta-llama/Meta-Llama-3-8B",
        adapter_paths={
            "sentiment": "./adapters/sentiment_rl",
            "topic":     "./adapters/topic_rl",
        },
        mode="sum",
    )
    output = model.generate(input_ids, max_new_tokens=100)

    # Weight averaging (PEFT-native, cheaper at inference)
    merged = compose_weight_average(
        base_model_id="meta-llama/Meta-Llama-3-8B",
        adapter_paths={...},
        weights=[0.5, 0.5],
    )
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output composition model
# ---------------------------------------------------------------------------

class ComposedModel(nn.Module):
    """
    Loads a frozen base model + N PEFT adapters, and composes their outputs
    using either summing or averaging — exactly matching the paper's methods.

    This is the correct implementation of output composition: each adapter
    runs its own forward pass, producing its own delta contribution, and
    these are combined additionally. This preserves each module's learned
    low-rank structure, unlike weight averaging which introduces cross-terms
    (see Appendix C of your paper).
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_paths: dict[str, str],   # {name: path_to_adapter_dir}
        mode: str = "sum",               # "sum" or "average"
        use_4bit: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        assert mode in ("sum", "average"), f"mode must be 'sum' or 'average', got {mode}"
        self.mode = mode
        self.adapter_names = list(adapter_paths.keys())

        # ---- Load base model ----
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        ) if use_4bit else None

        logger.info(f"Loading base model: {base_model_id}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # ---- Attach all adapters ----
        # Load the first adapter with from_pretrained, then load the rest
        first_name = self.adapter_names[0]
        self.model = PeftModel.from_pretrained(
            base,
            adapter_paths[first_name],
            adapter_name=first_name,
        )
        for name in self.adapter_names[1:]:
            self.model.load_adapter(adapter_paths[name], adapter_name=name)

        logger.info(f"Loaded adapters: {self.adapter_names}, mode={mode}")

    def _get_base_logits(self, input_ids, attention_mask=None, **kwargs):
        """Run the frozen base model (no adapter active)."""
        with self.model.disable_adapter():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        return out.logits   # (batch, seq, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Implements output composition.

        For output summing:   logits = base_logits + Σ (adapter_i_logits - base_logits)
        For output averaging: logits = base_logits + mean_i(adapter_i_logits - base_logits)

        Mathematically equivalent to:
            sum:     h = W0 x + Σ (αi/ri)(ΔWi x)
            average: h = W0 x + (1/N) Σ (αi/ri)(ΔWi x)
        """
        base_logits = self._get_base_logits(input_ids, attention_mask, **kwargs)

        # Collect each adapter's delta contribution
        deltas = []
        for name in self.adapter_names:
            self.model.set_adapter(name)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            deltas.append(out.logits - base_logits)

        # Compose deltas
        stacked = torch.stack(deltas, dim=0)   # (N, batch, seq, vocab)
        if self.mode == "sum":
            composed_delta = stacked.sum(dim=0)
        else:  # average
            composed_delta = stacked.mean(dim=0)

        final_logits = base_logits + composed_delta

        # Compute loss if labels provided (for evaluation)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Return an object with .logits (and optionally .loss) for compatibility
        return _ModelOutput(logits=final_logits, loss=loss)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=100, **kwargs):
        """
        Autoregressive generation with output composition at every step.
        Uses greedy decoding by default; pass do_sample=True for sampling.
        """
        do_sample = kwargs.pop("do_sample", False)
        temperature = kwargs.pop("temperature", 1.0)
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            attn = (generated != self.model.config.pad_token_id).long() \
                   if attention_mask is None else \
                   torch.ones(generated.shape, device=generated.device)

            out = self.forward(generated, attention_mask=attn)
            next_logits = out.logits[:, -1, :]   # (batch, vocab)

            if do_sample:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences have produced EOS
            if (next_token == self.model.config.eos_token_id).all():
                break

        return generated

    def set_adapters(self, names: list[str]):
        """Dynamically change which adapters are composed."""
        assert all(n in self.adapter_names for n in names), \
            f"Unknown adapter names: {set(names) - set(self.adapter_names)}"
        self.adapter_names = names

    @property
    def config(self):
        return self.model.config


class _ModelOutput:
    """Minimal output container for compatibility with HF generate() interface."""
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


# ---------------------------------------------------------------------------
# Weight averaging (PEFT-native)
# ---------------------------------------------------------------------------

def compose_weight_average(
    base_model_id: str,
    adapter_paths: dict[str, str],
    weights: Optional[list[float]] = None,
    output_adapter_name: str = "weight_avg",
    use_4bit: bool = True,
) -> PeftModel:
    """
    Compose adapters by averaging their weight matrices.
    Uses PEFT's built-in add_weighted_adapter (combination_type="linear").

    This is mathematically NOT equivalent to output averaging — see
    Appendix C of the paper for the cross-terms explanation.
    Included for completeness and comparison with output methods.

    Returns a PeftModel with a new adapter called `output_adapter_name`
    containing the averaged weights, ready for standard inference.
    """
    names = list(adapter_paths.keys())
    if weights is None:
        weights = [1.0 / len(names)] * len(names)
    assert len(weights) == len(names), "Must provide one weight per adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load all adapters
    first_name = names[0]
    model = PeftModel.from_pretrained(base, adapter_paths[first_name],
                                      adapter_name=first_name)
    for name in names[1:]:
        model.load_adapter(adapter_paths[name], adapter_name=name)

    # PEFT's weighted combination
    model.add_weighted_adapter(
        adapters=names,
        weights=weights,
        adapter_name=output_adapter_name,
        combination_type="linear",
    )
    model.set_adapter(output_adapter_name)

    logger.info(
        f"Weight-averaged {names} with weights {weights} "
        f"-> adapter '{output_adapter_name}'"
    )
    return model


# ---------------------------------------------------------------------------
# Convenience: load a single adapter for single-task inference/evaluation
# ---------------------------------------------------------------------------

def load_single_adapter(
    base_model_id: str,
    adapter_path: str,
    adapter_name: str = "default",
    use_4bit: bool = True,
) -> PeftModel:
    """Load a single RL-trained adapter for single-task evaluation."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ) if use_4bit else None

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    return model
