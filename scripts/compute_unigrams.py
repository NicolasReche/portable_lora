import os
import json
import math
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Loading dataset 'wikitext-103-raw-v1' as a general English corpus...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    print("Tokenizing and counting...")
    counts = Counter()
    total_tokens = 0
    
    def process_batch(batch):
        tokens = tokenizer(batch['text'], add_special_tokens=False)['input_ids']
        return {'tokens': tokens}
        
    tokenized = dataset.map(process_batch, batched=True, num_proc=4)
    
    for tokens in tqdm(tokenized['tokens'], desc="Counting"):
        counts.update(tokens)
        total_tokens += len(tokens)
        
    print(f"Total tokens processed: {total_tokens}")
    print(f"Unique tokens found: {len(counts)}")
    
    print("Computing log probabilities...")
    log_probs = {}
    for token_id, count in counts.items():
        log_probs[str(token_id)] = math.log(count / total_tokens)
        
    out_file = "data/llama3_unigram_log_probs.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(log_probs, f)
        
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
