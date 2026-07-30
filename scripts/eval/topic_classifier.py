from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Tuple
import numpy as np
from tqdm import tqdm


def set_seed(seed: int):
    """
    Set seeds for reproducibility.

    Args:
        seed (int): Seed value to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class TopicClassifier:
    def __init__(self, seed: int = None):
        """
        Initialize the topic classifier.

        Args:
            seed (int, optional): Seed for reproducibility
        """
        if seed is not None:
            set_seed(seed)

        model_name = "CohereForAI/c4ai-command-r-plus-4bit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    def predict(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        """
        Predict topics for a list of texts.

        Args:
            texts (List[str]): List of texts to classify

        Returns:
            Tuple[List[str], List[str]]: (processed_predictions, raw_predictions)
        """
        processed_predictions = []
        raw_predictions = []

        categories = "SPORTS, BUSINESS, WORLD, SCIENCE/TECHNOLOGY"
        system_message = "You are a helpful assistant that classifies texts by topic."
        user_message = """Classify the following text into one of these topic categories: {categories}.
Only reply with one of the possible topic categories. Do not include any other category or text.

{text}"""

        for text in tqdm(texts, desc="Classifying texts"):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message.format(categories=categories, text=text)}
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=5,
                    do_sample=False
                )

            # Get the full output including the prompt
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract only the model's response after the last CHATBOT_TOKEN
            raw_prediction = full_output.split("<|CHATBOT_TOKEN|>")[-1].strip()
            raw_predictions.append(raw_prediction)

            # Normalize prediction to match expected labels
            prediction = raw_prediction.upper().replace("<|END_OF_TURN_TOKEN|>", "")
            if "TECH" in prediction:
                prediction = "SCIENCE/TECHNOLOGY"
            elif "SPORTS" in prediction:
                prediction = "SPORTS"
            elif "BUSINESS" in prediction:
                prediction = "BUSINESS"
            elif "WORLD" in prediction:
                prediction = "WORLD"
            else:
                prediction = "WORLD"

            processed_predictions.append(prediction)

        return processed_predictions, raw_predictions


def load_topic_classifier(model_info: dict, device: str = None) -> TopicClassifier:
    """
    Load the topic classifier.

    Args:
        model_info (dict): Dictionary containing model information
        device (str): Device to load the model on

    Returns:
        TopicClassifier: The initialized classifier
    """
    return TopicClassifier(model_info['seed'])


# Register the classifier info
TOPIC_CLASSIFIER_INFO = {
    'model': 'cohere-topic-classifier',
    'loading_function': load_topic_classifier,
    'predict_function': lambda clf, texts: clf.predict(texts)
}
