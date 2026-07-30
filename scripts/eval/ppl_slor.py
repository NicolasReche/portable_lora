from typing import List, Tuple

import torch
from evaluate import load
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM


def load_gpt2_automodels(model_name: str) -> Tuple[object, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def load_causal_lm(model_name: str) -> Tuple[object, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def load_perplexity_model() -> object:
    return load("perplexity", module_type="metric")

def calculate_log_probability(input_ids, model):
    with torch.no_grad():
        outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()
    # collect the probability of the generated token
    # probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)
    log_probs = [log_prob.sum().item() for log_prob in gen_probs]
    return log_probs


def calculate_slor_batch(sequences: List[str],
                         metric_info: dict):
    tokenizer = metric_info['tokenizer']
    device = metric_info['device']
    language_model = metric_info['language_model']
    slors = []
    for sequence in sequences:
        if sequence.strip() == "" or sequence is None:
            slors.append(0)
            continue
        # Tokenize the input with BOS + sequence
        input_ids = tokenizer(
            tokenizer.bos_token + sequence,
            truncation=True, max_length=1024,
            return_tensors="pt"
        ).input_ids.to(device)

        # 1) Calculate log prob with language model
        log_prob_language_model = calculate_log_probability(input_ids, language_model)

        # 2) Calculate log prob with "unigram" model
        # Instead of `tokenizer.bos_token`, we should use the ID
        sos_token_id = tokenizer.bos_token_id

        # Build an input of shape [N, 2], where we pair [sos_token_id, next_token] for each token
        # in the original sequence
        # note: range(1, len(input_ids[0])) ensures we skip the first token (which is bos)
        # so that each pair is (BOS, next_token)
        pairs = []
        for i in range(1, len(input_ids[0])):
            pairs.append([sos_token_id, input_ids[0][i].item()])

        pairs_tensor = torch.tensor(pairs).to(device)

        # Now calculate log prob
        log_prob_unigram_model = sum(calculate_log_probability(pairs_tensor, language_model))

        # 3) SLOR calculation
        slor = log_prob_language_model[0] - log_prob_unigram_model
        slor /= (len(input_ids[0]) - 1)  # length normalization
        slors.append(slor)
    return slors


def compute_perplexity_batch(batch: List[str],
                             model_info: dict) -> dict:
    model_id = model_info['model']
    perplexity = model_info['perplexity']
    perplexity_results = {'overall': perplexity.compute(predictions=batch, model_id=model_id)}
    return perplexity_results['perplexities']
