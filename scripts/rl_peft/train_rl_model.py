import json
from functools import partial
import os
import argparse
import yaml
import wandb
import torch

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from reward_function import reward_function_v1, reward_function_v2, reward_function_v2_1, reward_function_v3, reward_function_v4

import random

def sample_to_prompt(sample: dict, attribute: str='sentiment'):
    """
    Extract the prompt for GRPO training and generate a contrast prompt.
    """
    prompt = f"[{attribute.upper()}] {sample['control']} [\\{attribute.upper()}] [ANS] {sample['input']}".strip()
    
    if attribute.lower() == 'sentiment':
        contrast_control = "Negative" if sample['control'].lower() == "positive" else "Positive"
    elif attribute.lower() == 'topic':
        topics = ["World", "Sports", "Business", "Sci/Tech"]
        if sample['control'] in topics:
            topics.remove(sample['control'])
        contrast_control = random.choice(topics)
    else:
        contrast_control = "None"
        
    contrast_prompt = f"[{attribute.upper()}] {contrast_control} [\\{attribute.upper()}] [ANS] {sample['input']}".strip()
    
    return {'prompt': prompt, 'contrast_prompts': contrast_prompt}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/training_config.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--run_name', type=str, default='sft_training_run')
    parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2', 'v2_1', 'v3', 'v4'], help='Reward function version to use')
    args = parser.parse_args()

    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['base_model']
    train_data_path = config['data']['train_file']
    eval_data_path = config['data']['val_file']
    output_path = os.path.join(args.model_dir, args.run_name)

    os.environ["WANDB_PROJECT"] = config['wandb_project']
    _ = wandb.init(
        project=config['wandb_project'],
        name=args.run_name,
        tags=[
            config['model']['base_model'].split('/')[-1],
            str(args.seed)])

    wandb.config.update({
        "train_dataset_name": config['data']['train_file'].split('/')[-1],
        "val_dataset_name": config['data']['val_file'].split('/')[-1],
        "control_attribute": ", ".join(config['attributes']),
    })

    train_dataset = load_dataset("csv", data_files=train_data_path)['train']

    eval_dataset = load_dataset("csv", data_files=eval_data_path)['train']
    print(train_dataset)
    print(eval_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    # Disable Qwen3 thinking mode — it prepends <think>...</think> blocks
    # before completions, which breaks completion_only_loss masking and
    # wastes context budget during training.
    if hasattr(tokenizer, "enable_thinking"):
        tokenizer.enable_thinking = False
 
    # Register CTG special tokens so the model learns to emit them as
    # single tokens rather than subword fragments. [\ANS] is used as
    # the generation stop token — the most critical one to get right.
    # attribute = config['attributes'][0]
    # ctg_special_tokens = ["[ANS]", r"[\ANS]"]
    # if attribute.upper() == "SENTIMENT":
    #     ctg_special_tokens += ["[SENTIMENT]", r"[\SENTIMENT]"]
    # elif attribute.upper() == "TOPIC":
    #     ctg_special_tokens += ["[TOPIC]", r"[\TOPIC]"]
    # tokenizer.add_special_tokens({"additional_special_tokens": ctg_special_tokens})
    # ans_close_id = tokenizer.convert_tokens_to_ids(r"[\ANS]")
    # print(f"CTG special tokens added: {ctg_special_tokens}")
    # print(f"[\\ANS] token id: {ans_close_id}")

    pad_token = config['model'].get('pad_token', None)
 
    if pad_token in tokenizer.get_vocab():
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
    elif pad_token:
        tokenizer.add_special_tokens({'pad_token': pad_token})
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
 
    tokenizer.padding_side = "right"  # right-padding for SFT; left-padding is for generation
    print(f"Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")


    #train_dataset = train_dataset.select(range(10))
    #eval_dataset = eval_dataset.select(range(10))

    train_dataset = train_dataset.map(sample_to_prompt, fn_kwargs={
        'attribute': config['attributes'][0],  # for now we only support one control attribute
    })
    eval_dataset = eval_dataset.map(sample_to_prompt, fn_kwargs={
        'attribute': config['attributes'][0],  # for now we only support one control attribute
    })

    bnb_config = None
    if 'quantization' in config:
        compute_dtype = getattr(torch, config['quantization']['bnb_4bit_compute_dtype'])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization']['use_4bit'],
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config['quantization']['use_nested_quant'],
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        # attn_implementation="flash_attention_2"
    )
 
    # Resize embeddings for the newly added CTG special tokens.
    # Must happen before kbit training prep and before LoRA is attached.
    # model.resize_token_embeddings(len(tokenizer))
 
    # # Register [\ANS] as an additional EOS so the model learns to stop there.
    # existing_eos = model.config.eos_token_id
    # if isinstance(existing_eos, int):
    #     existing_eos = [existing_eos]
    # elif existing_eos is None:
    #     existing_eos = []
    # if ans_close_id not in existing_eos:
    #     model.config.eos_token_id = existing_eos + [ans_close_id]
    # print(f"Model EOS token ids: {model.config.eos_token_id}")

    if bnb_config is not None and bnb_config.load_in_4bit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, config['model']['sft_adapter_path'], is_trainable=True)

    print(model)
    model.print_trainable_parameters()

    train_args = GRPOConfig(
        output_dir=output_path,
        seed=args.seed,
        data_seed=args.seed,
        max_prompt_length=config.get('max_prompt_length', 128),
        max_completion_length=config.get('max_seq_length', 1024) - config.get('max_prompt_length', 128),
        num_generations=config['training'].get('num_generations', 4),
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        optim=config['training']['optim'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        max_grad_norm=config['training']['max_grad_norm'],
        max_steps=config['training']['max_steps'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        save_strategy=config['training']['save_strategy'],
        save_total_limit=config['training']['save_total_limit'],
        save_steps=config['training']['save_steps'],
        eval_strategy=config['training'].get('eval_strategy', 'no'),
        eval_steps=config['training'].get('eval_steps', None),    # how often to evaluate
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],  # how often to log to W&B
        report_to="wandb",  # enable logging to W&B
        run_name=args.run_name,  # name of the W&B run (optional)
    )
    unigram_log_probs = None
    unigram_path = "data/llama3_unigram_log_probs.json"
    if os.path.exists(unigram_path):
        with open(unigram_path, "r", encoding="utf-8") as f:
            unigram_str = json.load(f)
            unigram_log_probs = {int(k): v for k, v in unigram_str.items()}
        print(f"Loaded unigram log probs from {unigram_path} (vocab size {len(unigram_log_probs)})")
    else:
        print(f"Warning: {unigram_path} not found. SLOR will fallback to uniform distribution.")

    if args.reward_version == 'v1':
        def selected_reward_func(prompts, completions, **kwargs):
            return reward_function_v1(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, **kwargs)
        selected_reward_func.__name__ = 'reward_function_v1'
    elif args.reward_version == 'v2':
        def selected_reward_func(prompts, completions, **kwargs):
            return reward_function_v2(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, **kwargs)
        selected_reward_func.__name__ = 'reward_function_v2'
    elif args.reward_version == 'v2_1':
        def selected_reward_func(prompts, completions, **kwargs):
            return reward_function_v2_1(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, **kwargs)
        selected_reward_func.__name__ = 'reward_function_v2_1'
    elif args.reward_version == 'v4':
        def selected_reward_func(prompts, completions, **kwargs):
            return reward_function_v4(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, unigram_log_probs=unigram_log_probs, **kwargs)
        selected_reward_func.__name__ = 'reward_function_v4'
    else:
        def selected_reward_func(prompts, completions, **kwargs):
            return reward_function_v3(prompts=prompts, completions=completions, model=model, tokenizer=tokenizer, unigram_log_probs=unigram_log_probs, **kwargs)
        selected_reward_func.__name__ = 'reward_function_v3'   

    trainer = GRPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if config['training'].get('eval_strategy', 'no') != 'no' else None,
        processing_class=tokenizer,
        args=train_args,
        reward_funcs=[selected_reward_func],
        
    )

    trainer.train()

