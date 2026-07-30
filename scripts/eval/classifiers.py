from typing import List, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from transformers import TextClassificationPipeline
from transformers import (DebertaV2Tokenizer, DebertaV2ForSequenceClassification,
                          AutoModelForSequenceClassification, AutoModelForCausalLM,
                          DistilBertTokenizer, DistilBertForSequenceClassification,
                          AutoTokenizer, AutoModelForSeq2SeqLM, GPT2Tokenizer,
                          GPT2LMHeadModel)

LABELS = {
    "POSITIVE": "positive",
    "NEGATIVE": "negative",
    "p": "positive",
    "n": "negative"
}


def load_hf_automodels(model: dict,
                       device: str="cuda:0") -> Tuple[object, object]:
    """
    Load a huggingface model and tokenizer with AutoTokenizer and 
    AutoModelForSeq2SeqLM classes respectively.

    Args:
        model (dict): dictionary representing the model to load in the format:
                        {
                            'model': model_name: str,
                            'tokenizer': tokenizer_name: str,
                            'loading_function': loading_function: Callable,
                            'predict_function': predict_function: Callable
                        }
        device (_type_, optional): device to load the model on. Defaults to "cuda:0".

    Returns:
        Tuple[object, object]: tokenizer, model
    """
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForSeq2SeqLM.from_pretrained(model['model']).to(device)
    return tokenizer, model

def load_hf_autoclassifier(model: dict,
                           device: str="cuda:0") -> Tuple[object, TextClassificationPipeline]:
    """
    Load a huggingface model and tokenizer with AutoTokenizer and
    AutoModelForSequenceClassification classes respectively.

    Args:
        model (dict): dictionary representing the model to load in the format:
                        {
                            'model': model_name: str,
                            'tokenizer': tokenizer_name: str,
                            'loading_function': loading_function: Callable,
                            'predict_function': predict_function: Callable
                        }
        device (_type_, optional): device to load the model on. Defaults to "cuda:0".

    Returns:
        Tuple[object, TextClassificationPipeline]: tokenizer, classifier
    """
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForSequenceClassification.from_pretrained(model['model']).to(device)
    classifier = TextClassificationPipeline(model=model,
                                            tokenizer=tokenizer,
                                            return_all_scores=False,
                                            device=0)
    return tokenizer, classifier

def load_prior_model(model: dict,
                     device: str="cuda:0") -> Tuple[object, object]:
    """
    Load model and tokenizer with DebertaV2Tokenizer and 
    DebertaV2ForSequenceClassification respectively
    from PriorCTG paper. #TODO: Add reference

    Args:
        model (dict): dictionary representing the model to load in the format:
                        {
                            'model': model_name: str,
                            'tokenizer': tokenizer_name: str,
                            'loading_function': loading_function: Callable,
                            'predict_function': predict_function: Callable
                        }
        device (_type_, optional): device to load the model on. Defaults to "cuda:0".

    Returns:
        Tuple[object, object]: tokenizer, model
    """
    tokenizer = DebertaV2Tokenizer.from_pretrained(model['tokenizer'])
    model = DebertaV2ForSequenceClassification.from_pretrained(model['model'],
                                                               num_labels=2).to(device)
    return tokenizer, model

def load_hf_distilbert_model(model: dict,
                             device: str="cuda:0") -> Tuple[object, object]:
    """
    Load model and tokenizer with DistilBertTokenizer and
    DistilBertForSequenceClassification respectively.

    Args:
        model (dict): dictionary representing the model to load in the format:
                        {
                            'model': model_name: str,
                            'tokenizer': tokenizer_name: str,
                            'loading_function': loading_function: Callable,
                            'predict_function': predict_function: Callable
                        }
        device (_type_, optional): device to load the model on. Defaults to "cuda:0".

    Returns:
        Tuple[object, object]: tokenizer, model
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model['tokenizer'])
    model = DistilBertForSequenceClassification.from_pretrained(model['model']).to(device)
    return tokenizer, model

def load_gpt2_automodels(model: dict,
                         device: str="cuda:0") -> Tuple[object, object]:
    tokenizer = GPT2Tokenizer.from_pretrained(model['tokenizer'])
    model = GPT2LMHeadModel.from_pretrained(model['model']).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def load_causal_lm(model: dict,
                   device: str="cuda:0") -> Tuple[object, object]:
    tokenizer = AutoTokenizer.from_pretrained(model['tokenizer'])
    model = AutoModelForCausalLM.from_pretrained(model['model']).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def get_batch_prediction_ditilbert_class(
        batch: List[str],
        tokenizer: DistilBertTokenizer,
        model: DistilBertForSequenceClassification,
        device: str="cuda:0") -> Tuple[List[str], List[list]]:
    """
    Get the predictions for a batch of texts using a DistilBert model classifier.

    Args:
        batch (List[str]): batch of texts to classify.
        tokenizer (DistilBertTokenizer): tokenizer for the model.
        model (DistilBertForSequenceClassification): classifier model.
        device (_type_, optional): device to run the model on. Defaults to "cuda:0".

    Returns:
        Tuple[List[str], List[list]]: list of predictions and logits.
    """
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    preds = []
    for sample in logits:
        predicted_class_id = sample.argmax().item()
        pred_class = model.config.id2label[predicted_class_id]
        preds.append(LABELS[pred_class])
    return preds, logits

def get_batch_pred_t5_class(batch: List[str],
                            tokenizer: AutoTokenizer,
                            model:AutoModelForSeq2SeqLM,
                            device: str="cuda:0") -> Tuple[List[str], None]:
    """
    Get the predictions for a batch of texts using a T5 model classifier.

    Args:
        batch (List[str]): batch of texts to classify.
        tokenizer (AutoTokenizer): tokenizer for the model.
        model (AutoModelForSeq2SeqLM): classifier model.
        device (_type_, optional): device to run the model on. Defaults to "cuda:0".

    Returns:
        Tuple[List[str], None]: list of predictions and None.
    """
    updated_batch = [f"sentiment: {text}" for text in batch]
    inputs = tokenizer(updated_batch,
                       return_tensors="pt",
                       padding=True,
                       truncation=True).input_ids.to(device)
    model_preds = model.generate(inputs, max_new_tokens=5)
    decoded_preds = tokenizer.batch_decode(sequences=model_preds, skip_special_tokens=True)
    preds = []
    for pred_class in decoded_preds:
        preds.append(LABELS[pred_class])
    return preds, None

def get_batch_pred_topic_pipeline(batch: List[str],
                                  tokenizer: AutoTokenizer,
                                  classifier: TextClassificationPipeline,
                                  device: str="cuda:0") -> Tuple[List[str], None]:
    """
    Get the predictions for a batch of texts using a pipeline classifier.

    Args:
        batch (List[str]): batch of texts to classify.
        tokenizer (AutoTokenizer): tokenizer for the model.
        classifier (TextClassificationPipeline): classifier model.
        device (str, optional): device to run the model on. Defaults to "cuda:0".

    Returns:
        Tuple[List[str], None]: list of predictions and None.
    """
    class_names = ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"]
    id2label = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    outputs = classifier(batch, **tokenizer_kwargs)
    preds = [id2label[class_names.index(o['label'])] for o in outputs]
    return preds, None


def get_batch_prediction_prior_class(
        batch: List[str],
        tokenizer: DebertaV2Tokenizer,
        model: DebertaV2ForSequenceClassification,
        device: str = "cuda:0"
) -> Tuple[List[str], torch.Tensor]:
    """
    Get the predictions for a batch of texts using a DebertaV2 model classifier.

    Args:
        batch (List[str]): batch of texts to classify.
        tokenizer (DebertaV2Tokenizer): tokenizer for the model.
        model (DebertaV2ForSequenceClassification): classifier model.
        device (str, optional): device to run the model on. Defaults to "cuda:0".

    Returns:
        Tuple[List[str], torch.Tensor]: list of predictions and logits.
    """
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    preds = []
    for sample in probs:
        predicted_class_id = sample.argmax().item()
        preds.append(predicted_class_id)
    return preds, probs


def prior_model_predict_sentiment(
        batch: List[str],
        tokenizer: DebertaV2Tokenizer,
        model: DebertaV2ForSequenceClassification,
        device: str = "cuda:0"
) -> Tuple[str, torch.Tensor]:
    id2label = {
        0: "negative",
        1: "positive"
    }
    preds, probs = get_batch_prediction_prior_class(batch, tokenizer, model, device)
    return [id2label[p] for p in preds], probs.cpu().numpy()


def prior_model_predict_topic(
        batch: List[str],
        tokenizer: DebertaV2Tokenizer,
        model: DebertaV2ForSequenceClassification,
        device: str = "cuda:0"
) -> Tuple[List[str], torch.Tensor]:
    """
    Predict the topic for each text in a batch of texts using a DebertaV2 model.

    Args:
        batch (List[str]): Texts to classify.
        model (DebertaV2ForSequenceClassification): Model to use for classification.
        tokenizer (DebertaV2Tokenizer): Tokenizer to use for tokenizing the texts.
        device (str, optional): Device to run the model on. Defaults to "cuda:0".

    Returns:
        Tuple[List[str], torch.Tensor]: List of predictions and logits.
    """
    topics = ['World', 'Sports', 'Business', 'Science/Technology']
    id2control_attribute = {
        0: "World",
        1: "Sports",
        2: "Business",
        3: "Science/Technology"
    }

    all_logits = {}
    all_preds = {}
    for top in topics:
        proc_texts = [f"{top}[SEP]{text}" for text in batch]

        all_preds[top], probs = get_batch_prediction_prior_class(proc_texts,
                                                                 tokenizer,
                                                                 model,
                                                                 device)
        all_logits[top] = [p[1] for p in probs.cpu().numpy()]
    final = np.array([all_logits[t] for t in topics]).T
    final_preds = np.array([all_preds[t] for t in topics]).T
    final_logits = final*final_preds
    wrap_preds = list(final_logits.argmax(-1))
    return [id2control_attribute[p] for p in wrap_preds], final_logits


CLASSIFIERS = {
    'sentiment': [
        {
            'model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'tokenizer': 'distilbert-base-uncased-finetuned-sst-2-english',
            'loading_function': load_hf_distilbert_model,
            'predict_function': get_batch_prediction_ditilbert_class
        },
        {
            'model': 'michelecafagna26/t5-base-finetuned-sst2-sentiment',
            'tokenizer': 'michelecafagna26/t5-base-finetuned-sst2-sentiment',
            'loading_function': load_hf_automodels,
            'predict_function': get_batch_pred_t5_class
        },
        {
            'model': 'models/evaluation/Yelp2-checkpoint-64000',
            'tokenizer': 'microsoft/deberta-v3-large',
            'loading_function': load_prior_model,
            'predict_function': prior_model_predict_sentiment
        }
    ],
    'topic': [
        {
            'model': 'textattack/distilbert-base-uncased-ag-news',
            'tokenizer': 'textattack/distilbert-base-uncased-ag-news',
            'loading_function': load_hf_autoclassifier,
            'predict_function': get_batch_pred_topic_pipeline
        },
        {
            'model': 'fabriceyhc/bert-base-uncased-ag_news',
            'tokenizer': 'fabriceyhc/bert-base-uncased-ag_news',
            'loading_function': load_hf_autoclassifier,
            'predict_function': get_batch_pred_topic_pipeline
        },
        {
            'model': './models/evaluation/AGnews-checkpoint-6000',
            'tokenizer': 'microsoft/deberta-v3-large',
            'loading_function': load_prior_model,
            'predict_function': prior_model_predict_topic
        }
    ],
    'toxicity': [
        {
            'model': 's-nlp/roberta_toxicity_classifier',
            'tokenizer': 's-nlp/roberta_toxicity_classifier',
            'loading_function': load_hf_autoclassifier,
            'predict_function': get_batch_pred_topic_pipeline
        },
        {
            'model': 'fabriceyhc/bert-base-uncased-ag_news',
            'tokenizer': 'fabriceyhc/bert-base-uncased-ag_news',
            'loading_function': load_hf_autoclassifier,
            'predict_function': get_batch_pred_topic_pipeline
        },
        {
            'model': './models/evaluation/',
            'tokenizer': 'microsoft/deberta-v3-large',
            'loading_function': load_prior_model,
            'predict_function': prior_model_predict_topic
        }
    ]
}
