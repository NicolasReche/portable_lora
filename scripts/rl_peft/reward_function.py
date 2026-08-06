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
            Where CE =  sentiment_CE || topic_CE
                    Diversity = (distinct_3 + distinct_2 + distinct_1) / 3
            

            - Definition of our CE :
                CE = exp(-Loss_CE(completion_tokens | control_prompt))
                (This represents the mathematical probability of generating the completion given the control tag)

            - Definition of our SLOR (Fluency):
                Instead of the traditional slow SLOR calculation, we use a fast and robust approximation (inverse perplexity):
                Fluency = exp(-Loss_CE(completion_tokens)) 

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
    print("--- 1. Testing Diversity Score ---")
    good_completion = "The sushi was fresh, delicious, and the service was fantastic!"
    bad_completion = "food food food food food food food food food"
    print("Good Completion Diversity:", diversity_score(good_completion))
    print("Bad Completion Diversity:", diversity_score(bad_completion))
    assert diversity_score(good_completion) > diversity_score(bad_completion)
    print("Diversity Unit Test Passed!\n")

    print("--- 2. Testing Model-based Scores (MOCK) ---")
    print("(Using a Mock Model to simulate the AI's loss without downloading anything...)")
    
    # We create fake objects that mimic HuggingFace Models and Tokenizers
    class MockTokenizer:
        def __call__(self, text, return_tensors="pt"):
            import torch
            length = 5 if text.startswith("[POSITIVE]") and "ANS" not in text else 15
            
            # Create a fake dictionary that supports the .to() method like a HuggingFace Tokenizer
            class FakeBatchEncoding(dict):
                def to(self, device):
                    return self
                    
            return FakeBatchEncoding({"input_ids": torch.zeros((1, length), dtype=torch.long)})
            
    class MockOutput:
        def __init__(self, loss_val):
            self.loss = torch.tensor(loss_val)
            
    class MockModel:
        def __init__(self):
            self.device = "cpu"
        def __call__(self, **kwargs):
            # We simulate the loss: a good text gives a low loss, a bad text gives a high loss
            labels = kwargs.get("labels")
            has_mask = (labels == -100).any().item()
            is_bad_completion = labels.shape[1] > 10 and kwargs.get("input_ids").sum() == 0 # Dummy condition
            
            # CE call
            if has_mask: 
                return MockOutput(0.5)
            # SLOR call
            else:
                return MockOutput(1.5)

    tokenizer = MockTokenizer()
    model = MockModel()

    prompt = "[POSITIVE] Yelp [\\POSITIVE] [ANS]"
    
    # Fake a good completion (Low loss = high math.exp(-loss))
    r_ce_good = control_effectiveness_score(prompt, good_completion, model, tokenizer)
    r_slor_good = fluency_score(good_completion, model, tokenizer)
    print(f"\n[Mocked Good Completion]")
    print(f"Simulated CE Loss: 0.5 -> Score: {r_ce_good:.4f}")
    print(f"Simulated SLOR Loss: 1.5 -> Score: {r_slor_good:.4f}")

    print("\n--- 3. Testing Total Reward Function (MOCK) ---")
    # To test the total loop, we just call the function. We expect it to calculate the correct weighted sum.
    rewards = reward_function_v1([prompt], [good_completion], model, tokenizer)
    
    # Manual verification:
    # Wce (0.45) * r_ce_good + Wslor (0.275) * r_slor_good + Wdiv (0.275) * div_good
    div_score = diversity_score(good_completion)
    expected = (0.45 * r_ce_good) + (0.275 * r_slor_good) + (0.275 * div_score)
    
    print(f"\nCalculated Reward: {rewards[0]:.4f}")
    print(f"Expected Reward:   {expected:.4f}")
    
    assert abs(rewards[0] - expected) < 1e-4
    print("Mocked Total Reward Test Passed! The mathematical combination logic works perfectly.")