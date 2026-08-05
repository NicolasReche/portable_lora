# Thinking about reward function
import torch
import math
from typing import List

def fluency_score(completion: str, model, tokenizer) -> float:
    """Approximates normalized SLOR fluency score"""
    if not completion.strip():
        return 0.0
        
    inputs = tokenizer(completion, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
        
    return math.exp(-loss)

def control_effectiveness_score(prompt: str, completion: str, model, tokenizer):
    """Computes CE using Negative Cross-Entropy Loss on completion tokens"""
    full_text = prompt + completion
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    labels = inputs["input_ids"].clone()
    
    # Mask out prompt tokens so loss is only calculated on completion
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()

    return math.exp(-loss)

def compute_distinct_n(text: str, n: int):
    """Calculates the Distinct-n metric for a given text"""
    tokens = text.strip().split()        
    if len(tokens) < n: return 0.0

    ngrams = []
    limit = len(tokens) - n + 1  # N+1 because we want to include the last N-1 tokens but not N-th token

    for i in range(limit):
        piece = tokens[i : i + n]
        ngrams.append(tuple(piece))

    return len(set(ngrams)) / len(ngrams)

def diversity_score(completion: str):
    """Averages distinct-1, distinct-2, and distinct-3"""
    d1 = compute_distinct_n(completion, 1)
    d2 = compute_distinct_n(completion, 2)
    d3 = compute_distinct_n(completion, 3)
    return (d1 + d2 + d3) / 3.0

def reward_function_v1(prompts: List[str], completions: List[str], model, tokenizer):
    """
    Reward function design :
            - reward = Wce * CE + Wslor * SLOR + Wdiv * diversity
            With Wce > 0, Wslor > 0, Wdiv > 0, and sum(Wce, Wslor, Wdiv) = 1
            Where Wce,Wslor,Wdiv are the weigth for each reward function
            Where CE = (sentiment + topic)/2
                    Diversity = (distinct_3 + distinct_2 + distinct_1) / 3
            

            - Definition of our CE :
                for the sentiement task :
                    CE_sentiment = - Loss_CE(completion_tokens | sentiment_prompt)
                for the topic task :
                    CE_topic = - Loss_CE(completion_tokens | topic_prompt)
                CE = (CE_sentiment + CE_topic) / 2

            - Definition of our SLOR (Fluency):
                SLOR = (ln P_LM(completion) - ln P_unigram(completion)) / length(completion)
                Normalized: SLOR_norm = max(0.0, SLOR / 10.0)

            - Definition of our Diversity:
                distinct_n = count(unique_ngrams) / count(total_ngrams)
                Diversity = (distinct_1 + distinct_2 + distinct_3) / 3

            - Reward combination :
                Wce = 0.45
                Wslor = 0.275
                Wdiv =  0.275           

    """
    Wce, Wslor, Wdiv = 0.45, 0.275, 0.275
    rewards = []

    for prompt, completion in zip(prompts, completions):
        r_ce = control_effectiveness_score(prompt, completion, model, tokenizer)
        r_slor = fluency_score(completion, model, tokenizer)
        r_div = diversity_score(completion)

        total_reward = Wce * r_ce + Wslor * r_slor + Wdiv * r_div
        rewards.append(float(total_reward))

    return rewards

if __name__ == "__main__":
    # 1. Test Dummy Strings on Diversity
    good_completion = "The sushi was fresh, delicious, and the service was fantastic!"
    bad_completion = "food food food food food food food food food"
    print("Good Completion Diversity:", diversity_score(good_completion)) # High (~1.0)
    print("Bad Completion Diversity:", diversity_score(bad_completion))   # Low (~0.1)

    # Output verification
    assert diversity_score(good_completion) > diversity_score(bad_completion)
    print("Diversity Unit Test Passed!")