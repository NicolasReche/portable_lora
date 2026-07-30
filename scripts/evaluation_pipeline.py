import os
from collections import defaultdict, Counter
from typing import List, Dict, Callable, Tuple

import json
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

from eval.distinct_n import eval_distinct
from eval.classifiers import CLASSIFIERS
from eval.ppl_slor import (
    calculate_slor_batch,
    load_perplexity_model,
    compute_perplexity_batch,
    load_gpt2_automodels,
    load_causal_lm)


class ClassifierWrapper:
    def __init__(self,
                 model_info: dict,
                 load_function: Callable[[Dict], Tuple],
                 prediction_function: Callable[[List[str]], Tuple[List[str], np.ndarray]]):
        """
        Wrapper class for classifiers.

        Args:
            model_info (dict): The info to load the proper classifier model and tokenizer.
            load_function (Callable[[Dict], Tuple]): Function to load or instantiate the classifier.
            prediction_function (Callable[[List[str]], Tuple[List[str], np.ndarray]]): 
                                    Function to predict a class for each text in the list.
                                    The function must take a list of strings (texts) as input.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_info = model_info
        self.load_function = load_function
        self.tokenizer, self.model = self.load_function(model_info, device=device)
        self.predict = prediction_function

    def get_classifier_name(self) -> str:
        """
        Get the name of the classifier.

        Returns:
            str: The name of the classifier.
        """
        return self.model_info.get('model', 'UnknownClassifier')

    def predict_in_batches(self,
                           texts: List[str],
                           batch_size: int=32) -> Tuple[List[str], np.ndarray]:
        """
        Make predictions in batches.

        Args:
            texts (List[str]): List of input texts.
            batch_size (int, optional): Batch size for predictions. Defaults to 32.

        Returns:
            Tuple[List[str], np.ndarray]: Tuple of predictions and logits
        """
        all_predictions = []
        all_logits = []
        cl_name = self.get_classifier_name()
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting with {cl_name}"):
            batch_texts = texts[i:i + batch_size]
            preds, logits = self.predict(batch_texts, self.tokenizer, self.model)
            all_predictions.extend(preds)
            #all_logits.extend(logits.cpu().numpy())
        return all_predictions, np.array(all_logits)


class EvaluationPipeline:
    def __init__(self,
                 metrics: List[str],
                 classifier_infos: List[Dict]=None,
                 batch_size: int=32):
        """
        Class for evaluating classifiers and calculating metrics.

        Args:
            metrics (List[str]): List of metrics to evaluate.
            classifier_infos (List[Dict], optional): List of dictionaries containing information to
                                                    load each classifier. Defaults to None.
            batch_size (int, optional): Batch size for predictions. Defaults to 32.
        """
        self.batch_size = batch_size
        self.classifiers = self._initialize_classifiers(classifier_infos)
        self.metrics = self._get_metrics(metrics)
        self._load_models(metrics)

    def _initialize_classifiers(self, classifier_infos: List[Dict]) -> List[ClassifierWrapper]:
        """
        Initialize classifier wrappers from classifier information.

        Args:
            classifier_infos (List[Dict]): List of dictionaries containing information to
                                            load each classifier.

        Returns:
            List[ClassifierWrapper]: List of ClassifierWrapper instances
        """
        if classifier_infos is None:
            return []
        return [
            ClassifierWrapper(
                model_info=info,
                load_function=info['loading_function'],
                prediction_function=info['predict_function']
            ) for info in classifier_infos
        ]

    def _get_metrics(self, metrics: List[str]) -> Dict[str, Callable]:
        """
        Returns a dictionary of metric functions based on user desired metrics.

        Args:
            metrics (List[str]): list of metrics to evaluate.

        Returns:
            Dict[str, Callable]: dictionary of metric functions.
        """
        avail_metrics = {
            'accuracy': {
                'compute_function': accuracy_score
            },
            'precision': {
                'compute_function': precision_score
            },
            'recall': {
                'compute_function': recall_score
            },
            'f1': {
                'compute_function': f1_score
            },
            'distinct-n': {
                'compute_function': eval_distinct
            },
            'slor': {
                'models': [],
                'compute_function': calculate_slor_batch
            },
            'perplexity': {
                'models': [],
                'compute_function': compute_perplexity_batch
            }
        }
        return {metric: avail_metrics[metric] for metric in metrics if metric in avail_metrics}

    def _load_models(self, metrics: Dict):
        """
        Load models required for specific metrics such as SLOR and Perplexity.

        Args:
            metrics (Dict): Dictionary of metrics to evaluate.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if 'slor' in metrics:
            # Load multiple SLOR models
            slor_models_info = []
            tokenizer, model = load_gpt2_automodels("openai-community/gpt2-xl")
            slor_models_info.append({
                'model': 'openai-community/gpt2-xl',
                'language_model': model,
                'tokenizer': tokenizer,
                'device': device})
            tokenizer, model = load_causal_lm("bigscience/bloom-1b7")
            slor_models_info.append({
                'model': 'bigscience/bloom-1b7',
                'language_model': model,
                'tokenizer': tokenizer,
                'device': device})

            for model_info in slor_models_info:
                self.metrics['slor']['models'].append(model_info)

        if 'perplexity' in metrics:
            # Load multiple Perplexity models
            perplexity_models_info = [
                {'model': 'openai-community/gpt2-xl',
                 'perplexity': load_perplexity_model(),
                 'device': device},
                {'model': "bigscience/bloom-1b7",
                 'perplexity': load_perplexity_model(),
                 'device': device}
            ]
            for model_info in perplexity_models_info:
                self.metrics['perplexity']['models'].append(model_info)

    def compute_batch_metric(self,
                              texts: List[str],
                              metric_name: str,
                              metric_info: Dict) -> dict:
        """
        Generalized function to compute a metric in batches (e.g., SLOR or Perplexity).

        Args:
            texts (List[str]): List of input texts
            metric_name (str): Name of the metric to compute
            metric_info (Dict): List of metric scores

        Returns:
            dict: Dictionary of scores for each model
        """
        scores = defaultdict(list)
        compute_function = metric_info['compute_function']
        for model_info in metric_info['models']:
            model_name = model_info['model']
            for i in tqdm(range(0, len(texts), self.batch_size),
                          desc=f"Computing {metric_name} with {model_name}"):
                batch_texts = texts[i:i + self.batch_size]
                batch_scores = compute_function(batch_texts, model_info)
                scores[model_name].extend(batch_scores)
        return scores

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all classifiers on the provided texts and calculate specified metrics.
        Additionally, calculate metrics for subsets of the data based on the class of 
        the ground truth labels.

        Args:
            texts (List[str]): List of input texts
            labels (List[str]): List of ground truth labels

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of evaluation results
        """
        results = defaultdict(lambda: defaultdict(dict))
        predictions = defaultdict(dict)  # To store predictions for future use

        # Get the predictions for each classifier if the metric requires it
        if any(metric in self.metrics for metric in ['accuracy', 'precision', 'recall']):
            for clf_wrapper in self.classifiers:
                clf_name = clf_wrapper.get_classifier_name()

                # Generate predictions
                y_pred, logits = clf_wrapper.predict_in_batches(texts, self.batch_size)
                predictions[clf_name]['predictions'] = y_pred
                predictions[clf_name]['logits'] = logits.tolist()  # Save logits for future use
                y_pred = [pred.lower() for pred in y_pred]

                # Overall metrics
                for metric_name, metric_info in self.metrics.items():
                    if metric_name in ['accuracy', 'precision', 'recall']:
                        # Overall metrics
                        if metric_name == 'accuracy':
                            score = metric_info['compute_function'](labels, y_pred)
                        else:
                            score = metric_info['compute_function'](labels,
                                                                    y_pred,
                                                                    average='weighted')
                        results[metric_name][clf_name]['overall'] = score

                        # Metrics for each class subset (e.g., sentiment-specific metrics)
                        unique_classes = np.unique(labels)
                        for cls in unique_classes:
                            cls_indices = [i for i, label in enumerate(labels) if label == cls]
                            y_test_subset = [labels[i] for i in cls_indices]
                            y_pred_subset = [y_pred[i] for i in cls_indices]
                            if metric_name == 'accuracy':
                                score = metric_info['compute_function'](y_test_subset,
                                                                        y_pred_subset)
                            else:
                                score = metric_info['compute_function'](y_test_subset,
                                                                        y_pred_subset,
                                                                        average='weighted')
                            results[metric_name][clf_name][f'class_{cls}'] = score

            # compute the average of the metrics over all classifiers
            for metric_name, metric_info in self.metrics.items():
                if metric_name in ['accuracy', 'precision', 'recall']:
                    scores = [results[metric_name][clf_name]['overall']
                              for clf_name in results[metric_name]]
                    results[metric_name]['overall'] = np.mean(scores)

                    # Metrics for each class subset
                    unique_classes = np.unique(labels)
                    for cls in unique_classes:
                        clf_names = [clf_wrapper.get_classifier_name()
                                     for clf_wrapper in self.classifiers]
                        scores = [results[metric_name][clf_name][f'class_{cls}']
                                  for clf_name in clf_names]
                        results[metric_name][f'overall_{cls}'] = np.mean(scores)

        # Distinct-n metric calculation per class subset
        if 'distinct-n' in self.metrics:
            unique_classes = np.unique(labels)
            for cls in unique_classes:
                cls_indices = [i for i, label in enumerate(labels) if label == cls]
                text_subset = [texts[i] for i in cls_indices]
                distinct_n_score = eval_distinct(text_subset)
                results['distinct-n'][f'class_{cls}'] = distinct_n_score
            results['distinct-n']['overall'] = np.mean(
                [results['distinct-n'][f'class_{cls}'] for cls in unique_classes])

        # Compute SLOR, Perplexity, and other metrics
        for metric_name, metric_info in self.metrics.items():
            if metric_name in ['slor', 'perplexity']:
                metric_scores = self.compute_batch_metric(texts, metric_name, metric_info)
                overall = []
                for model_name, scores in metric_scores.items():
                    results[metric_name][model_name] = {
                        'overall': np.mean(scores)
                    }
                    overall.extend(scores)
                results[metric_name]['overall'] = np.mean(overall)
            # Execute other metrics that were not computed before and do not require
            # predictions or batches
            elif metric_name not in ['accuracy', 'precision', 'recall', 'f1', 'distinct-n']:
                results[metric_name]['overall'] = metric_info['compute_function'](labels, texts)
                # Metrics for each class subset
                unique_classes = np.unique(labels)
                for cls in unique_classes:
                    cls_indices = [i for i, label in enumerate(labels) if label == cls]
                    y_test_subset = [labels[i] for i in cls_indices]
                    text_subset = [texts[i] for i in cls_indices]
                    results[metric_name][f'class_{cls}'] = metric_info['compute_function'](
                        y_test_subset,
                        text_subset
                    )
        return results, predictions

def run_evaluation_pipeline(metrics: List[str],
                            texts: List[str],
                            labels: List[str],
                            classifier_infos: List[Dict]=None) -> Tuple[Dict, Dict]:
    """
    Run the evaluation pipeline with the specified classifiers and metrics.
    If the JSON file at save_path already exists, load the existing metric results and then
    add the new metric results from this execution.

    Args:
        metrics (List[str]): List of metrics to evaluate.
        texts (List[str]): Texts to evaluate.
        labels (List[str]): Ground truth labels.
        save_path (str, optional): Path to save evaluation results. Defaults to "results.json".
        predictions_path (str, optional): Path to save predictions labels.
                                            Defaults to None.
        classifier_infos (List[Dict], optional): List of dictionaries containing information 
                                                to load each classifier. Defaults to None.

    Returns:
        Tuple[Dict, Dict]: The evaluation results and predictions of the classifiers.
    """
    # Initialize and run the evaluation pipeline
    pipeline = EvaluationPipeline(metrics, classifier_infos, batch_size=1)
    results, predictions = pipeline.evaluate(texts, labels)

    return results, predictions

def evaluate(config: dict, outputs_path: str="results.json", predictions_path: str=None):
    """
    Evaluate the model using the evaluation pipeline and the given config.

    Args:
        config (dict): Configuration dictionary containing all necessary information for the
                        evaluation.
    """

    # load the texts
    with open(outputs_path, 'r', encoding="utf-8") as f:
        outputs = json.load(f)
    
    print(outputs.keys())
    all_predictions = {}
    for key in outputs['test_sets']:
        print(f"Evaluating on test set: {key}")

        raw_texts = outputs['test_sets'][key]['generated_texts']
        print(f"Number of texts: {len(raw_texts)}")

        labels = []
        texts = []
        for item in raw_texts:
            t = item['prompt'] + ' ' + item['completion']
            t = t.split("[ANS]")[-1].strip()  # Extract the generated part of the text
            t = t.split("[\ANS]")[0].strip()  # Remove any trailing special tokens

            texts.append(t)
            labels.append(item['label'].lower())

        if 'max_words' in config:
            texts = [' '.join(text.split(" ")[:config['max_words']]) for text in texts]

        # run the evaluation pipeline
        metrics = config['evaluation']['metrics']
        attribute = config['attribute']
        predictions = {}

        classifier_infos = CLASSIFIERS[attribute]
        results, predictions = run_evaluation_pipeline(metrics,
                                            texts,
                                            labels,
                                            classifier_infos=classifier_infos)

        outputs['test_sets'][key]['evaluation_results'] = results
        # Save results to a file
        with open(outputs_path, 'w', encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)

        # Save predictions to a file
        if predictions_path is not None:
            all_predictions[key] = predictions
            with open(predictions_path, 'w', encoding="utf-8") as f:
                json.dump(all_predictions, f, indent=4)

    print("Evaluation complete")
