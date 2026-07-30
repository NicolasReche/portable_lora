# General imports
from dataclasses import dataclass, field

import yaml

from evaluation_pipeline import evaluate

# Hugging Face imports
# models
from transformers import HfArgumentParser

from utils import set_all_seed


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, 
    what their capacity and features are, and what size model you want to train.
    """
    config: str = field(metadata={
        "help": "Config file containing all variables and hyperparameters in YAML format."
    })
    output_file: str = field(default="results.json", metadata={
        "help": "File to save the evaluation results to."
    })
    predictions_file: str = field(default="predictions.json", metadata={
        "help": "File to save the predictions to."
    })
    seed: int = field(default=42, metadata={
        "help": "Seed for reproducibility."
    })


if __name__ == '__main__':
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load config file
    with open(script_args.config, 'r', encoding="utf-8") as config_f:
        config = yaml.safe_load(config_f)

    set_all_seed(script_args.seed)

    if "evaluation" in config:
        print("Evaluating")
        evaluate(
            config,
            outputs_path=script_args.output_file,
            predictions_path=script_args.predictions_file)
        # if multiple attribute control is being evaluate call multiple control evaluation
