import numpy as np
from math import log, ceil
from functools import reduce


def normalize_signal(
        sig: np.ndarray
) -> np.ndarray:
    """Normalizes a signal to the proper range.

    Args:
        sig: An array-like input signal.

    Returns:
        array-like: The normalized signal.
    """
    try:
        intinfo = np.iinfo(sig.dtype)
        return sig / max(intinfo.max, -intinfo.min)

    except ValueError:  # array is not integer dtype
        return sig / max(sig.max(), -sig.min())


def resample_signal(
    sig: np.ndarray,
    native_sr: int,
    target_sr: int,
    dtype: str = "float32"
) -> np.ndarray:
    """Resamples an audio signal to the target sampling rate.

    This function prioritizes computational accuracy by performing the resampling
    internally using the highest precision available (usually float64). However,
    the final result is cast to the specified `dtype` to ensure the output matches
    the desired format.

    Args:
        sig: The input audio signal as a NumPy array.
        native_sr: The original sample rate of the signal (samples per second).
        target_sr: The desired sample rate of the resampled signal.
        d_type: The desired data type for the resampled signal (default: float32).

    Returns:
        The resampled signal as a NumPy array with the specified `dtype`.
    """
    
    duration = len(sig) / native_sr
    n_target_samples = int(duration * target_sr)

    # Generate time arrays (default to float64 for maximum precision)
    time_x_source = np.linspace(0, duration, len(sig))
    time_x = np.linspace(0, duration, n_target_samples)

    # Perform resampling (np.interp will use float64 internally)
    resampled_buffer = np.interp(time_x, time_x_source, sig)

    # Cast resampled buffer to desired data type
    resampled_buffer = resampled_buffer.astype(dtype)

    return resampled_buffer


# TODO: Make it also work for (n_samples, ) arrays for robustness
def resample_2d(
        audio_data: np.ndarray,
        native_sr: int,
        target_sr: int,
        dtype: str = "float32"
) -> np.ndarray:
    """Resamples 2D audio data (multi-channel) to a target sampling rate.

    Args:
        audio_data: The input audio data as a 2D numpy (n_samples, n_channels).
        native_sr: The native sampling rate of the input audio data.
        target_sr: The target sampling rate as integer.

    Returns:
        np.ndarray: The resampled audio data as a 2D numpy.ndarray.
        
    Raises:
        ValueError: If the input `audio_data` is not a 2-dimensional NumPy array.
    """
    
    # Check if audio data is 2D
    if audio_data.ndim != 2:
        raise ValueError(
            "Input audio data must be a 2-dimensional NumPy array "
            "(n_samples, n_channels)."
        )

    # Initialize a list to hold resampled channels
    resampled_channels = []

    # Iterate through each channel in the audio data
    for i in range(audio_data.shape[1]):
        channel_data = audio_data[:, i]
        resampled_channel_data = resample_signal(
            channel_data, native_sr, target_sr, dtype
        )

        resampled_channels.append(resampled_channel_data)

    # Stack the resampled channels back into a 2D array
    resampled_audio_data = np.stack(resampled_channels, axis=1)

    return resampled_audio_data


def trim_signal(
        array: np.ndarray,
        axis=0,
        epsilon: float = 1e-5
) -> np.ndarray:
    """Trims the noise from beginning and end of a signal.

    Args:
      array: The input signal.
      axis: The axis to trim.
      epsilon: The max value to be considered as noise.
    Returns:
      np.ndarray: A signal of start and stop with shape `[..., 2, ...]`.
    """
    shape = array.shape
    length = shape[axis]

    nonzero = np.greater(array, epsilon)
    check = np.any(nonzero, axis=axis)

    forward = np.array(nonzero, np.int64)
    reverse = forward[::-1]

    start = np.where(check, np.argmax(forward, axis=axis), length)
    stop = np.where(check, np.argmax(reverse, axis=axis),
                    np.array(0, np.int64))
    stop = length - stop

    return array[start:stop]


def create_chunks(
        array: np.ndarray,
        chunk_length: int
) -> list[np.ndarray]:
    """

    Args:
        array:
        chunk_length:

    Returns:

    """
    n_chunks = ceil(len(array) / float(chunk_length))
    return [array[i * chunk_length:(i + 1) * chunk_length]
            for i in range(int(n_chunks))]


def dBFS(
        array: np.ndarray,
        sample_width: int
) -> float:
    """Calculates the decibels relative to full scale (dBFS) of an audio.

    Args:
        array:
        sample_width:

    Returns:

    """
    from caits.fe._statistical import rms_value
    rms = rms_value(array)
    if not rms:
        return -float("infinity")
    return ratio_to_db(rms / max_possible_amplitude(sample_width))


def max_possible_amplitude(
        sample_width: int
) -> float:
    bits = sample_width * 8
    max_possible_val = (2 ** bits)

    # since half is above 0 and half is below the max amplitude is divided
    return max_possible_val / 2


def ratio_to_db(ratio, val2=None, using_amplitude=True):
    """
    Converts the input float to db, which represents the equivalent
    to the ratio in power represented by the multiplier passed in.
    """
    ratio = float(ratio)

    # accept 2 values and use the ratio of val1 to val2
    if val2 is not None:
        ratio = ratio / val2

    # special case for multiply-by-zero (convert to silence)
    if ratio == 0:
        return -float('inf')

    if using_amplitude:
        return 20 * log(ratio, 10)
    else:  # using power
        return 10 * log(ratio, 10)
