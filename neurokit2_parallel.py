# This file attempts to replicate the
# neurokit2.ecg_process and ecg_interval_related methods,
# but vectorized to support multi-lead ECGs without loops.
import numpy as np
import pandas as pd
import scipy
import scipy.signal


def ecg_clean(ecg_signal, sampling_rate=500):
    """
    signal, np.array. shape should be (number of leads, signal length)
    """

    # Remove slow drift with highpass Butterworth.
    sos = scipy.signal.butter(
        5, [0.5,], btype="highpass", output="sos", fs=sampling_rate
    )
    clean = scipy.signal.sosfiltfilt(sos, ecg_signal, axis=0).T

    # DC offset removal with 50hz powerline filter (convolve average kernel)
    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / 50))
    else:
        b = np.ones(2)
    a = [
        len(b),
    ]
    clean = scipy.signal.filtfilt(b, a, clean, method="pad", axis=1).T

    return clean


def ecg_peaks(ecg_signal, sampling_rate=500):
    """
    parallelized version of nk.ecg_peaks(method='neurokit', correct_artifacts=True)
    """
    pass

    # nk.ecg_findpeaks()


def _ecg_findpeaks_neurokit(
    signal,
    sampling_rate=1000,
    smoothwindow=0.1,
    avgwindow=0.75,
    gradthreshweight=1.5,
    minlenweight=0.4,
    mindelay=0.3,
):
    """All tune-able parameters are specified as keyword arguments.

    The `signal` must be the highpass-filtered raw ECG with a lowcut of .5 Hz.

    """
    # Compute the ECG's gradient as well as the gradient threshold. Run with
    # show=True in order to get an idea of the threshold.
    grad = np.gradient(signal)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))

    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs = np.where(np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:]))[0]
    end_qrs = np.where(np.logical_and(qrs[0:-1], np.logical_not(qrs[1:])))[0]
    # Throw out QRS-ends that precede first QRS-start.
    end_qrs = end_qrs[end_qrs > beg_qrs[0]]

    # Identify R-peaks within QRS (ignore QRS that are too short).
    num_qrs = min(beg_qrs.size, end_qrs.size)
    min_len = np.mean(end_qrs[:num_qrs] - beg_qrs[:num_qrs]) * minlenweight
    peaks = [0]

    for i in range(num_qrs):

        beg = beg_qrs[i]
        end = end_qrs[i]
        len_qrs = end - beg

        if len_qrs < min_len:
            continue

        # Find local maxima and their prominence within QRS.
        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks.
            if peak - peaks[-1] > mindelay:
                peaks.append(peak)

    peaks.pop(0)

    peaks = np.asarray(peaks).astype(int)  # Convert to int
    return peaks


def signal_smooth(signal, kernel="boxcar", size=10, alpha=0.1):
    if isinstance(signal, pd.Series):
        signal = signal.values

    # Get window.
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    # Extend signal edges to avoid boundary effects.
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))

    # Compute moving average.
    smoothed = np.convolve(w, x, mode="same")
    smoothed = smoothed[size:-size]

    return smoothed
