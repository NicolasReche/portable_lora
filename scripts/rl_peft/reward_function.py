import math
import torch
from typing import List, Dict, Optional

def get_contrast_prompt(prompt: str) -> str:
    """Derives contrasting prompt for sentiment and topic tasks."""
    p_lower = prompt.lower()
    if "positive" in p_lower:
        return prompt.replace("positive", "negative").replace("Positive", "Negative")
    elif "negative" in p_lower:
        return prompt.replace("negative", "positive").replace("Negative", "Positive")
    elif "world" in p_lower:
        return prompt.replace("world", "sports").replace("World", "Sports")
    elif "sports" in p_lower:
        return prompt.replace("sports", "world").replace("Sports", "World")
    elif "business" in p_lower:
        return prompt.replace("business", "sci/tech").replace("Business", "Sci/Tech")
    elif "sci/tech" in p_lower or "technology" in p_lower:
        return prompt.replace("sci/tech", "business").replace("Sci/Tech", "Business")
    return prompt

def diversity_score(text: str) -> float:
    tokens = text.strip().split()
    if not tokens:
        return 0.0
    distinct_1 = len(set(tokens)) / len(tokens)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else distinct_1
    trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
    distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else distinct_2
    return (distinct_1 + distinct_2 + distinct_3) / 3.0

def fluency_score(completion: str, model, tokenizer) -> float:
    if not completion.strip():
        return 0.0
    inputs = tokenizer(completion, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    return math.exp(-min(loss, 20.0))

def contrastive_control_effectiveness_score(target_prompt: str, contrast_prompt: str, completion: str, model, tokenizer) -> float:
    def get_log_prob(prompt: str) -> float:
        full_text = prompt + completion
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            num_tokens = labels.shape[1] - prompt_len
        return -loss * num_tokens

    log_p_target = get_log_prob(target_prompt)
    log_p_contrast = get_log_prob(contrast_prompt)
    diff = max(min(log_p_target - log_p_contrast, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-diff))

def slor_score(completion: str, model, tokenizer, unigram_log_probs: Optional[Dict[int, float]] = None) -> float:
    if not completion.strip():
        return 0.0
    inputs = tokenizer(completion, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    
    input_ids = inputs["input_ids"][0].tolist()
    N = max(len(input_ids), 1)
    
    if unigram_log_probs is not None:
        log_p_unigram = sum(unigram_log_probs.get(tok, -12.0) for tok in input_ids)
    else:
        # Uniform vocab fallback if no empirical dictionary provided
        vocab_size = getattr(tokenizer, "vocab_size", 128256)
        log_p_unigram = -math.log(vocab_size) * N

    log_p_lm = -loss * N
    slor = (log_p_lm - log_p_unigram) / N
    # Normalize with sigmoid to keep reward in [0, 1]
    return 1.0 / (1.0 + math.exp(-max(min(slor, 20.0), -20.0)))

def reward_function_v1(prompts: List[str], completions: List[str], model, tokenizer, **kwargs) -> List[float]:
    rewards = []
    for prompt, comp in zip(prompts, completions):
        r_ce = fluency_score(prompt + comp, model, tokenizer)
        r_fl = fluency_score(comp, model, tokenizer)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_fl + 0.275 * r_div)
    return rewards

def reward_function_v2(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, **kwargs) -> List[float]:
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_effectiveness_score(prompt, cp, comp, model, tokenizer)
        r_fl = fluency_score(comp, model, tokenizer)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_fl + 0.275 * r_div)
    return rewards

def reward_function_v2_1(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, **kwargs) -> List[float]:
    return reward_function_v2(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, contrast_prompts=contrast_prompts, **kwargs)

def reward_function_v3(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, unigram_log_probs: Optional[Dict[int, float]] = None, **kwargs) -> List[float]:
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_effectiveness_score(prompt, cp, comp, model, tokenizer)
        r_slor = slor_score(comp, model, tokenizer, unigram_log_probs)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_slor + 0.275 * r_div)
    return rewards

def reward_function_v4(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, **kwargs) -> List[float]:
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_effectiveness_score(prompt, cp, comp, model, tokenizer)
        r_fl = fluency_score(comp, model, tokenizer)
        tokens = comp.strip().split()
        if tokens:
            from collections import Counter
            counts = Counter(tokens)
            probs = [c / len(tokens) for c in counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs)
            max_ent = math.log2(len(tokens)) if len(tokens) > 1 else 1.0
            r_entropy = entropy / max_ent if max_ent > 0 else 1.0
        else:
            r_entropy = 0.0
        rewards.append(0.45 * r_ce + 0.275 * r_fl + 0.275 * r_entropy)
    return rewards
