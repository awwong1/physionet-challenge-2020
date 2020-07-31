# This file attempts to replicate the
# neurokit2.ecg_process and ecg_interval_related methods,
# but vectorized to support multi-lead ECGs without loops.
import functools

import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
import scipy.signal

ECG_LEAD_NAMES = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)


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
    signals_info = map(
        functools.partial(
            _ecg_peaks_partial,
            sampling_rate=sampling_rate,
            lead_rpeaks=lead_rpeaks,
            ecg_signal=ecg_signal,
            ecg_lead_names=ecg_lead_names,
        ),
        enumerate(lead_rpeaks),
    )

    signals, info = zip(*signals_info)
    signals = pd.concat(signals)
    return signals, info


def _ecg_peaks_partial(
    lead_idx_rpeaks,
    sampling_rate=500,
    lead_rpeaks=None,
    ecg_signal=None,
    ecg_lead_names=None,
):
    lead_idx, rpeaks = lead_idx_rpeaks

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

    return instant_peaks, rpeaks_info


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


def signal_rate(all_r_peaks, sampling_rate=500, desired_length=None):
    """
    somewhat parallelized version of nk.signal_rate(interpolation_method="monotone_cubic")

    all_r_peaks: infos from ecg_peaks return
    """

    rate = dict(
        map(
            functools.partial(
                _signal_rate_partial,
                sampling_rate=sampling_rate,
                desired_length=desired_length,
            ),
            enumerate(all_r_peaks),
        )
    )

    return rate


def _signal_rate_partial(kv, sampling_rate=500, desired_length=None):
    k, v = kv
    peaks = v["ECG_R_Peaks"]

    # Sanity checks.
    if len(peaks) < 3:
        # needs at least 3 peaks to compute rate, otherwise NaN
        return k, np.full(desired_length, np.nan)

    # edge case if peaks desired length request is larger than max peak index
    while desired_length <= peaks[-1]:
        peaks = peaks[:-1]

    if len(peaks) < 3:
        # needs at least 3 peaks to compute rate, otherwise NaN
        return k, np.full(desired_length, np.nan)

    # Calculate period in sec, based on peak to peak difference and make sure
    # that rate has the same number of elements as peaks (important for
    # interpolation later) by prepending the mean of all periods.
    period = np.ediff1d(peaks, to_begin=0) / sampling_rate
    period[0] = np.mean(period[1:])

    # Interpolate all statistics to desired length.
    if desired_length is not None:
        x_new = np.arange(desired_length)

        period = scipy.interpolate.PchipInterpolator(peaks, period, extrapolate=True)(
            x_new
        )

        # Swap out the cubic extrapolation of out-of-bounds segments generated by
        # scipy.interpolate.PchipInterpolator for constant extrapolation akin to the behavior of
        # scipy.interpolate.interp1d with fill_value=([period[0]], [period[-1]].
        period[: peaks[0]] = period[peaks[0]]
        period[peaks[-1] :] = period[peaks[-1]]  # noqa: E203

    return k, 60 / period


def ecg_quality(ecg_signal, all_r_peaks, sampling_rate=500):
    """somewhat parallelized version of nk.ecg_quality
    all_r_peaks: infos from ecg_peaks return
    """

    # Get heartbeats
    heartbeats = ecg_segment(ecg_cleaned, rpeaks, sampling_rate)
    data = epochs_to_df(heartbeats).pivot(index="Label", columns="Time", values="Signal")
    data.index = data.index.astype(int)
    data = data.sort_index()

    # Filter Nans
    missing = data.T.isnull().sum().values
    nonmissing = np.where(missing == 0)[0]

    data = data.iloc[nonmissing, :]

    # Compute distance
    dist = distance(data, method="mean")
    dist = rescale(np.abs(dist), to=[0, 1])
    dist = np.abs(dist - 1)  # So that 1 is top quality

    # Replace missing by 0
    quality = np.zeros(len(heartbeats))
    quality[nonmissing] = dist

    # Interpolate
    quality = signal_interpolate(rpeaks, quality, x_new=np.arange(len(ecg_cleaned)), method="quadratic")

    return quality



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