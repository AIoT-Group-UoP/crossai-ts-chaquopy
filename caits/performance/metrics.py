import numpy as np
from typing import List, Union, Optional, Any
TensorLike = Union[np.ndarray, Any]


def compute_class(predictions: TensorLike) -> np.ndarray:
    return np.argmax(np.mean(predictions, axis=0), axis=1)


def compute_entropy(predictions: TensorLike) -> np.ndarray:
    epsilon = np.finfo(float).eps
    return np.mean(
        -np.sum(
            predictions * np.log(predictions + epsilon), axis=2
        ) / np.log(2), axis=0
    )


# Global dictionary of metric functions
_stats_functions = {
    "class": compute_class,
    "probas": lambda x: x,
    "mean_pred": lambda x: np.mean(x, axis=0),
    "std": lambda x: np.std(x, axis=0),
    "variance": lambda x: np.var(x, axis=0),
    "entropy": compute_entropy,
}


def prediction_statistics(
        probabilities: TensorLike,
        stats: Optional[Union[List[str], str]] = "all"
) -> dict:
    """Analyzes prediction probabilities to assess model trustworthiness
    and training adequacy. This function computes statistics from prediction
    probabilities, assuming probabilities have the shape
    (n_repeats, n_instances, n_classes). The analysis includes class
    predictions, raw probabilities, and metrics like mean, standard deviation,
    variance, and entropy across multiple prediction repeats. High variability
    (e.g., high standard deviation) in the prediction probabilities suggests
    less reliability in the model's predictions, indicating areas where the
    model may require further training or adjustment.

    Args:
        probabilities: A 3D array of shape (n_repeats, n_instances, n_classes)
                       containing prediction probabilities.
        stats: Specifies the types of results to compute. Options include:
                 "class" for class predictions, "probas" for returning the raw
                 probabilities, "mean_pred" for the  mean probabilities across
                 repeats, "std" for standard deviation, "variance" for
                 variance, "entropy" for entropy, and "all" for all metrics.

    Returns:
        A dictionary containing the computed statistics from the prediction
        probalities according to the `stats` option.
    """
    if stats == "all":
        metrics = list(_stats_functions.keys())
    if isinstance(metrics, str):
        metrics = [metrics]

    results = {}
    for metric in metrics:
        if metric in _stats_functions:
            results[metric] = _stats_functions[metric](probabilities)
        else:
            print(f"Metric '{metric}' not recognized.")

    return results
