from typing import Any, Optional, Union

import numpy as np
from scipy.interpolate import interp1d

TensorLike = Union[np.ndarray, Any]


def interpolate_probas(
        probabilities: np.ndarray,
        sampling_rate: int,
        Ws: float,
        n_points: int = None,
        kind: Optional[str] = "cubic",
        clamp: Optional[bool] = True
) -> np.ndarray:
    """Interpolates prediction probabilities for a smoother representation
    over time or samples.

    This function applies cubic interpolation to a matrix of class
    probabilities associated with instances or window segments in a
    time series, allowing for the visualization of smoothed probability
    transitions across classes.

    Args:
        probabilities: A 2D array where each row represents an
            instance (or window) and each column a class probability.
        sampling_rate: The sampling rate of the time series data, used to
            scale the interpolation points in relation to the original data.
        Ws: The window size in seconds, defining the temporal span of each
            instance or window in the original time series.
        n_points: The number of points for the interpolated output.
            If None, calculated as `n_instances * sampling_rate * Ws`.
        kind: The type of interpolation. Deafults to `linear`.
        clamp: Flag to clamp interpolated values between range [0, 1].
               Defaults to False.

    Returns:
        np.ndarray: A 2D array of shape `(n_points, n_classes)` containing the
            interpolated probabilities for each class. This array offers a
            continuous and smooth representation of class probabilities
            over time.
    """
    # Determine the shape of the input probability matrix
    n_instances, n_classes = probabilities.shape

    # Automatically calculate the number of interpolation points
    # if not provided. Goal here is the number of points to match the
    # original length of the time series instance
    if n_points is None:
        n_points = int(n_instances * sampling_rate * Ws)

    # Initialize the output array for interpolated probabilities
    interpolated_probabilities = np.zeros((n_points, n_classes))

    # Prepare the original and target x-values for interpolation
    x_original = np.arange(n_instances)
    x_interpolated = np.linspace(0, n_instances - 1, n_points)

    # Perform cubic interpolation for each class
    for i in range(n_classes):
        # Create the interpolator function for the current class
        interpolator = interp1d(x_original, probabilities[:, i],
                                kind=kind, fill_value='extrapolate')

        # Apply interpolation and store the results
        interpolated_probs = interpolator(x_interpolated)

        if clamp:
            # Clamp the interpolated values to [0, 1]
            interpolated_probs = np.clip(interpolated_probs, 0, 1)

        # Apply interpolation and store the results
        interpolated_probabilities[:, i] = interpolated_probs

    return interpolated_probabilities


def get_gt_events_from_dict(
        events: dict,
        class_names: list[str],
        sr: int = None
) -> dict:
    """Extracts and optionally converts start and end intervals from a given
    JSON structure to samples. The output is a dictionary keyed by the original
    keys from `events`, with values being lists of tuples. Each tuple
    represents an interval, including start and end points, and the label.

    Args:
        events: The JSON data containing the intervals.
        sr: Sampling rate to convert start and end from time to samples.
            If None, the values are used as is.

    Returns:
        dict: A dictionary where each key corresponds to the original keys
              in `events`, and each value is a list of tuples. Each tuple
              contains the start, end, and the label of an interval.
    """
    intervals = {
        key: [
            (
                int(item["start"] * sr) if item.get("type") == "time" else item["start"],
                int(item["end"] * sr) if item.get("type") == "time" else item["end"],
                list(class_names).index(item["label"])
            ) for item in items
        ] for key, items in events.items()
    }

    return intervals
