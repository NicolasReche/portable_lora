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
    """Averages distinct-1, distinct-2, and distinct-3"""
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
    """Approximates normalized SLOR fluency score"""
    if not completion.strip():
        return 0.0
    inputs = tokenizer(completion, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    return math.exp(-min(loss, 20.0))

def contrastive_control_effectiveness_score(target_prompt: str, contrast_prompt: str, completion: str, model, tokenizer) -> float:
    """
    Computes contrastive control score:
    sigmoid(log P(completion | target_prompt) - log P(completion | contrast_prompt))
    """
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

def contrastive_control_normalized_score(target_prompt: str, contrast_prompt: str, completion: str, model, tokenizer) -> float:
    """
    Computes per-token length-normalized contrastive control score:
    sigmoid(mean_loss_contrast - mean_loss_target)
    """
    def get_mean_loss(prompt: str) -> float:
        full_text = prompt + completion
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        labels = inputs["input_ids"].clone()

        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            mean_loss = outputs.loss.item()
        return mean_loss

    mean_loss_target = get_mean_loss(target_prompt)
    mean_loss_contrast = get_mean_loss(contrast_prompt)
    diff = max(min(mean_loss_contrast - mean_loss_target, 50.0), -50.0)
    return 1.0 / (1.0 + math.exp(-diff))

def slor_score(completion: str, model, tokenizer, unigram_log_probs: Optional[Dict[int, float]] = None) -> float:
    """
    Computes true SLOR using unigram_log_probs
    SLOR = (1/N) * (log P_LM(x) - log P_unigram(x))
    """
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
    """
    Reward function V1 (Baseline Control + Proxy Fluency):
        reward = Wce * R_control_ce + Wslor * R_fluency + Wdiv * R_diversity

    1. Control Reward (R_control_ce):
       - R_control = exp(-Loss_CE)
       - Basic control effectiveness based on raw conditional probability.

    2. Fluency (Proxy):
       - Fluency = exp(-Loss_CE(completion_tokens | target_prompt))
       - Measures basic text probability under the base model.

    3. Diversity:
       - Diversity = (distinct_1 + distinct_2 + distinct_3) / 3

    4. Weights:
       - Wce = 0.45
       - Wslor = 0.275
       - Wdiv = 0.275
    """
    rewards = []
    for prompt, comp in zip(prompts, completions):
        r_ce = fluency_score(prompt + comp, model, tokenizer)
        r_fl = fluency_score(comp, model, tokenizer)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_fl + 0.275 * r_div)
    return rewards

def reward_function_v2(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, **kwargs) -> List[float]:
    """
    Reward function V2 (Contrastive Control):
        reward = Wce * R_control_contrastive + Wslor * R_fluency + Wdiv * R_diversity

    Changes from V1:
        1. Control Reward (R_control_contrastive):
           - No longer raw exp(-Loss_CE).
           - R_control = sigmoid(log P(completion | target_prompt) - log P(completion | contrast_prompt))
           - Isolates true attribute alignment by canceling out general token frequency/fluency biases.

    Same as V1:
        2. Fluency (Proxy):
           - Fluency = exp(-Loss_CE(completion_tokens | target_prompt)) 
           - Measures basic text probability under the base model.

        3. Diversity:
           - Diversity = (distinct_1 + distinct_2 + distinct_3) / 3

        4. Weights:
           - Wce = 0.45
           - Wslor = 0.275
           - Wdiv =  0.275
    """
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
    """
    Reward function V2_1 (Length-Normalized Contrastive Control):
    reward = Wce * R_control_normalized + Wslor * R_fluency + Wdiv * R_diversity
    """
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_normalized_score(prompt, cp, comp, model, tokenizer)
        r_fl = fluency_score(comp, model, tokenizer)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_fl + 0.275 * r_div)
    return rewards

def reward_function_v3(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, unigram_log_probs: Optional[Dict[int, float]] = None, **kwargs) -> List[float]:
    """
    Reward function V3 (Length-Normalized Contrastive Control + SLOR Fluency):
        reward = Wce * R_control_normalized + Wslor * R_SLOR + Wdiv * R_diversity

    Changes from V2_1:
        1. Fluency (SLOR):
           - Replaces the inverse perplexity proxy with true SLOR normalization.
           - SLOR = (1/N) * (log P_LM(x) - log P_unigram(x))
           - Prevents penalizing rare, complex, or domain-specific words.

    Same as V2_1:
        2. Control Reward (R_control_normalized):
           - R_control = sigmoid(mean_loss_contrast - mean_loss_target)

        3. Diversity:
           - Diversity = (distinct_1 + distinct_2 + distinct_3) / 3

        4. Weights:
           - Wce = 0.45
           - Wslor = 0.275
           - Wdiv = 0.275
    """
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_normalized_score(prompt, cp, comp, model, tokenizer)
        r_slor = slor_score(comp, model, tokenizer, unigram_log_probs)
        r_div = diversity_score(comp)
        rewards.append(0.45 * r_ce + 0.275 * r_slor + 0.275 * r_div)
    return rewards

def reward_function_v4(prompts: List[str], completions: List[str], model, tokenizer, contrast_prompts: Optional[List[str]] = None, unigram_log_probs: Optional[Dict[int, float]] = None, **kwargs) -> List[float]:
    """
    Reward function V4 (Length-Normalized Contrastive Control + SLOR Fluency + Shannon Entropy Diversity):
        reward = Wce * R_control_normalized + Wslor * R_SLOR + Wdiv * R_entropy

    Changes from V3:
        1. Diversity (Shannon Entropy):
           - Penalizes repetition by computing the distribution of unigrams (tokens).
           - R_entropy = normalized_entropy(tokens)

    Same as V3:
        2. Control Reward (R_control_normalized):
           - R_control = sigmoid(mean_loss_contrast - mean_loss_target)

        3. Fluency (SLOR):
           - SLOR = (1/N) * (log P_LM(x) - log P_unigram(x))
           
        4. Weights:
           - Wce = 0.45
           - Wslor = 0.275
           - Wdiv = 0.275
    """
    if contrast_prompts is None:
        contrast_prompts = [get_contrast_prompt(p) for p in prompts]
    rewards = []
    for prompt, cp, comp in zip(prompts, contrast_prompts, completions):
        r_ce = contrastive_control_normalized_score(prompt, cp, comp, model, tokenizer)
        r_slor = slor_score(comp, model, tokenizer, unigram_log_probs)
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
        rewards.append(0.45 * r_ce + 0.275 * r_slor + 0.275 * r_entropy)
    return rewards
