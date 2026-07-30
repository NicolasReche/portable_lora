"""
This module contains utility functions for the project.

Functions:
    - set_seed(seed: int) -> None: set the seed to torch for reproducibility.
    - postprocess_function(output: str,
                            prompt: str,
                            final_tag: str=None, 
                            sos_token: str='<s>',
                            eos_token: str='</s>') -> str: 
                                postprocess an output removing the prompt, consider the output only 
                                until the final_tag token (if not None) and removing sos and eos 
                                tokens.
    - postprocess_function_batch(outputs: List[str], 
                                prompts: List[str],
                                final_tag: str=None,
                                sos_token: str='<s>',
                                eos_token: str='</s>') -> List[str]: postprocess a batch of outputs.
"""

import json
from typing import List

import wandb

import torch

from transformers import set_seed


TARGET_MODULES = {
    'llama3': "all-linear",
    'mistral': [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
}

def check_wandb_project_exists(project_name: str) -> str:
    api = wandb.Api()

    projects = api.projects()

    for proj in projects:
        if proj.name == project_name:
            return True
    return False

def get_wandb_run_id_by_name(project_name: str, run_name: str) -> str:
    """
    Retrieve a WandB run ID based on the run name.

    Args:
        project_name (str): The name of the project in WandB.
        run_name (str): The name of the run to retrieve.

    Returns:
        str: The run ID of the desired run.
    """
    if not check_wandb_project_exists(project_name):
        return None

    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{project_name}")

    # Loop through runs to find the matching run name
    for run in runs:
        if run.name == run_name and run.state == "finished":
            return run.id
    return None


def load_json(filepath: str, encoding: str="utf-8") -> dict:
    """
    Load a json object from a file.

    Args:
        filepath (_type_): filepath of the file to load.
        encoding (str, optional): Encoding of the file. Defaults to "utf-8".

    Returns:
        dict: json object loaded from the file.
    """
    with open(filepath, "r", encoding=encoding) as file:
        return json.load(file)


def save_json(data: dict, filepath: str, encoding: str="utf-8"):
    """
    Save a json object to a file.

    Args:
        data (dict | List[dict]): json object to save.
        filepath (str): filepath of the file to save.
        encoding (str, optional): encoding of the file. Defaults to "utf-8".
    """
    with open(filepath, 'w', encoding=encoding) as res_file:
        json.dump(data, res_file)


def get_target_modules(model_name: str) -> List[str]:
    """
    Get the target modules for the specified model.

    Args:
        model_name (str): name of the model.

    Returns:
        List[str] | str: list of target modules or a single target module.
    """
    model_name = "mistral" if "Mistral" in model_name else model_name
    model_name = "llama3" if "Llama-3" in model_name else model_name
    return TARGET_MODULES[model_name] if model_name in TARGET_MODULES else "all-linear"


def set_all_seed(seed: int) -> None:
    """
    Set the seed to torch for reproducibility.

    Args:
        seed (int): seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def postprocess_function(output: str,
                         prompt: str,
                         final_tag: str=None,
                         sos_token: str='<s>',
                         eos_token: str='</s>') -> str:
    """
    Postprocess an output removing the prompt, consider the output only until the 
    final_tag token (if not None) and removing sos and eos tokens.

    Args:
        output (str): text to postprocess.
        prompt (str): prompt to remove from the output.
        final_tag (str, optional): special token that identifies the end of the output. 
                                    Defaults to None.
        sos_token (str, optional): start of sentence special token. Defaults to '<s>'.
        eos_token (str, optional): end of sentence special token. Defaults to '</s>'.

    Returns:
        str: postprocessed output.
    """
    output = output.replace(prompt, "")

    start_prompt = prompt.split('[ANS]')[-1].strip()
    output = ' '.join([start_prompt, output])

    if final_tag is not None:
        output = output.split(final_tag)[0]
    if sos_token is not None:
        output = output.replace(sos_token, '')
    if eos_token is not None:
        output = output.replace(eos_token, '')

    output = output.replace(']', '').replace('[', '').replace('\n', ' ').strip()
    return output


def postprocess_function_batch(outputs: List[str],
                               prompts: List[str],
                               final_tag: str=None,
                               sos_token: str='<s>',
                               eos_token: str='</s>') -> List[str]:
    """
    Postprocess a batch of outputs.

    Args:
        outputs (List[str]): outputs to postprocess.
        prompts (List[str]): prompts to remove from the outputs.
        final_tag (str, optional): special token that identifies the end of the output. 
                                    Defaults to None.
        sos_token (str, optional): start of sentence special token. Defaults to '<s>'.
        eos_token (str, optional): end of sentence special token. Defaults to '</s>'.

    Returns:
        List[str]: list of the postprocessed outputs.
    """
    final_outputs = []
    for i, prompt in enumerate(prompts):
        output = outputs[i]
        output = postprocess_function(output,
                                      prompt,
                                      final_tag=final_tag,
                                      sos_token=sos_token,
                                      eos_token=eos_token)
        final_outputs.append(output)
    return final_outputs
