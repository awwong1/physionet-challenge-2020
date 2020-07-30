# This file attempts to replicate the
# neurokit2.ecg_process and ecg_interval_related methods,
# but vectorized to support multi-lead ECGs without loops.
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import scipy.signal

ECG_LEAD_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def ecg_clean(ecg_signal, sampling_rate=500):
    """
    parallelized version of nk.ecg_clean(method="neurokit")
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


def ecg_peaks(ecg_signal, sampling_rate=500, ecg_lead_names=ECG_LEAD_NAMES):
    """
    somewhat parallelized version of nk.ecg_peaks(method='neurokit', correct_artifacts=True)
    
    each lead may have different dimensioned peaks anyways,
    so parallelization of the peak fixing/detection is not trivial
    """

    # nk.ecg_findpeaks()
    lead_rpeaks = _ecg_findpeaks_neurokit(ecg_signal, sampling_rate=sampling_rate)

    signals = []
    info = {}

    # correct artifacts
    for lead_idx, rpeaks in enumerate(lead_rpeaks):
        _, rpeaks = nk.signal_fixpeaks(
            {"ECG_R_Peaks": rpeaks},
            sampling_rate=sampling_rate,
            iterative=True,
            method="Kubios",
        )
        lead_rpeaks[lead_idx] = rpeaks
        rpeaks_info = {"ECG_R_Peaks": rpeaks}

        # nk.signal_formatpeaks()
        if len(rpeaks) > 0:
            instant_peaks = nk.signal_formatpeaks(
                rpeaks_info, desired_length=len(ecg_signal), peak_indices=rpeaks_info
            )
        else:
            instant_peaks = pd.DataFrame({"ECG_R_Peaks": [0.0,] * len(ecg_signal)})
        instant_peaks["ECG_Sig_Name"] = ecg_lead_names[lead_idx]
        signals.append(instant_peaks)
        info[lead_idx] = rpeaks_info

    signals = pd.concat(signals)
    return signals, info


def signal_rate(peaks, sampling_rate=500, desired_length=None):
    """
    somewhat parallelized version of nk.signal_rate(interpolation_method="monotone_cubic")

    peaks: info dictionary from ecg_peaks return
    """

    # Sanity checks.
    if len(peaks) <= 3:
        print(
            "NeuroKit warning: _signal_formatpeaks(): too few peaks detected"
            " to compute the rate. Returning empty vector."
        )
        return np.full(desired_length, np.nan)

    # Calculate period in sec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length != np.size(peaks):
        period = signal_interpolate(peaks, period, x_new=np.arange(desired_length), method=interpolation_method)

    rate = 60 / period

    return rate

    pass


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
    grad = np.gradient(signal, axis=0)
    absgrad = np.abs(grad)
    smooth_kernel = int(np.rint(smoothwindow * sampling_rate))
    avg_kernel = int(np.rint(avgwindow * sampling_rate))
    smoothgrad = _signal_smooth(absgrad, kernel="boxcar", size=smooth_kernel)
    avggrad = _signal_smooth(smoothgrad, kernel="boxcar", size=avg_kernel)
    gradthreshold = gradthreshweight * avggrad
    mindelay = int(np.rint(sampling_rate * mindelay))

    # Identify start and end of QRS complexes.
    qrs = smoothgrad > gradthreshold
    beg_qrs, beg_qrs_leads = np.where(
        np.logical_and(np.logical_not(qrs[0:-1]), qrs[1:])
    )
    end_qrs, end_qrs_leads = np.where(
        np.logical_and(qrs[0:-1], np.logical_not(qrs[1:]))
    )

    # group indices per lead index
    beg_qrs_map = {}
    for sig_idx, lead_idx in zip(beg_qrs, beg_qrs_leads):
        sig_idxs = beg_qrs_map.get(lead_idx, np.empty(0, dtype=np.int64))
        beg_qrs_map[lead_idx] = np.append(sig_idxs, sig_idx)

    end_qrs_map = {}
    for sig_idx, lead_idx in zip(end_qrs, end_qrs_leads):
        sig_idxs = end_qrs_map.get(lead_idx, np.empty(0, dtype=np.int64))
        end_qrs_map[lead_idx] = np.append(sig_idxs, sig_idx)

    signal_len, num_leads = signal.shape

    lead_peaks = []
    for lead_idx in range(num_leads):
        beg_qrs = beg_qrs_map.get(lead_idx, np.zeros(1))
        end_qrs = end_qrs_map.get(lead_idx, np.zeros(1))

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
            data = signal[beg:end, lead_idx]
            locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

            if locmax.size > 0:
                # Identify most prominent local maximum.
                peak = beg + locmax[np.argmax(props["prominences"])]
                # Enforce minimum delay between peaks.
                if peak - peaks[-1] > mindelay:
                    peaks.append(peak)

        peaks.pop(0)

        lead_peaks.append(np.asarray(peaks).astype(int))  # Convert to int
    return lead_peaks


def _signal_smooth(signal, kernel="boxcar", size=10, alpha=0.1):
    if isinstance(signal, pd.Series):
        signal = signal.values

    # Get window.
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()

    sig_len, num_leads = signal.shape

    # Extend signal edges to avoid boundary effects.
    x = np.concatenate(
        (
            signal[0] * np.ones((size, num_leads)),
            signal,
            signal[-1] * np.ones((size, num_leads)),
        )
    )

    # Compute moving average.
    smoothed = scipy.signal.convolve(x, w[:, np.newaxis], mode="same")
    smoothed = smoothed[size:-size]

    return smoothed
