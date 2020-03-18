"""Functions for manual feature extraction of ECG 12 lead signals.
"""
import math
import re
from functools import partial
from collections import OrderedDict

import numpy as np
import scipy
from biosppy.signals.ecg import (
    christov_segmenter,
    compare_segmentation,
    correct_rpeaks,
    engzee_segmenter,
    extract_heartbeats,
    gamboa_segmenter,
    hamilton_segmenter,
    ssf_segmenter,
)
from biosppy.signals.tools import filter_signal, smoother
from biosppy.utils import ReturnTuple
from scipy.signal import find_peaks, resample, windows
from sklearn.neighbors import KernelDensity
from wfdb import Record
from wfdb.io import _header

LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
SEX = ("Male", "Female")
RPEAK_DET_ALG = ("hamilton", "engzee", "christov", "gamboa", "ssf", "consensus")


def consensus(lists_of_rpeaks, length=8500, bandwidth=1.0):
    """
    var_list: iterable of iterables. sub-iterables must be sorted in ascending order.
    """
    X = np.array([i for sub in lists_of_rpeaks for i in sub])[:, np.newaxis]
    X_plot = np.linspace(0, length, length)[:, np.newaxis]

    kde = KernelDensity(bandwidth=bandwidth).fit(X)
    log_dens = kde.score_samples(X_plot)

    peaks, _ = find_peaks(log_dens)

    return np.array(tuple(int(i) for i in X_plot[peaks].squeeze()))


def extract_ecg_features(signal, sampling_rate=500):
    """Reimplemented biosppy's ECG signal feature extraction algorithm.
    https://github.com/PIA-Group/BioSPPy/blob/e65da30f6379852ecb98f8e2e0c9b4b5175416c3/biosppy/signals/ecg.py#L28

    The provided algorithm processes one signal at a time.
    signal: type(numpy.ndarray), shape of (leads, signal_length)
    """
    # Original approach...
    #
    #     features = tuple(
    #         dict(x)
    #         for x in map(partial(ecg, sampling_rate=sampling_rate, show=False), signal)
    #     )
    #
    # that threw a `ValueError: Not enough beats to compute heart rate`
    # Therefore, do a heuristic approach for vetting beat classification?
    # Additional work done to speed up multi-lead analysis
    #
    # some bad files:
    # A0115.mat
    # A0718.mat
    # A3762.mat

    sampling_rate = float(sampling_rate)
    num_leads, length = signal.shape

    if num_leads > 12:
        raise RuntimeError(
            f"Maximum of 12 leads supported, signal contains {num_leads}"
        )

    # filter signal
    order = int(0.3 * sampling_rate)

    def filter_map(sig):
        return filter_signal(
            signal=sig,
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=[3, 45],
            sampling_rate=sampling_rate,
        )["signal"]

    filtered = tuple(map(filter_map, signal))

    # segment
    def segment_map(sig):
        return hamilton_segmenter(signal=sig, sampling_rate=sampling_rate)["rpeaks"]

    rpeaks = list(map(segment_map, filtered))
    rpeak_det = ["hamilton"] * len(rpeaks)

    # SANITY CHECK! If a a lead returns <2 peaks, it's an error!
    # try a different segmentation algorithm?
    ensure_rpeaks_valid(rpeaks, length, filtered, sampling_rate, rpeak_det)

    # correct R-peak locations
    def correct_map(sig_rpeak):
        sig, rpeak = sig_rpeak
        return correct_rpeaks(
            signal=sig, rpeaks=rpeak, sampling_rate=sampling_rate, tol=0.05
        )["rpeaks"]

    rpeaks = tuple(map(correct_map, zip(filtered, rpeaks)))

    # extract templates
    def extract_map(sig_rpeak):
        sig, rpeak = sig_rpeak
        return extract_heartbeats(
            signal=sig,
            rpeaks=rpeak,
            sampling_rate=sampling_rate,
            before=0.2,
            after=0.4,
        )

    templates, rpeaks = zip(*(map(extract_map, zip(filtered, rpeaks))))

    # compute heart rate over time
    def hr_map(beats):
        """Compute instantaneous heart rate from an array of beat indices.

        Parameters
        ----------
        beats : array
            Beat location indices.
        sampling_rate : int, float, optional
            Sampling frequency (Hz).
        smooth : bool, optional
            If True, perform smoothing on the resulting heart rate.
        size : int, optional
            Size of smoothing window; ignored if `smooth` is False.

        Returns
        -------
        index : array
            Heart rate location indices.
        heart_rate : array
            Instantaneous heart rate (bpm).

        Notes
        -----
        * Assumes normal human heart rate to be between 40 and 200 bpm.

        """
        # check inputs
        if beats is None:
            raise TypeError("Please specify the input beat indices.")

        if len(beats) < 2:
            raise ValueError("Not enough beats to compute heart rate.")

        # compute heart rate
        ts = beats[1:]
        hr = sampling_rate * (60.0 / np.diff(beats))

        # smooth with moving average
        if len(hr) > 1:
            hr, _ = smoother(signal=hr, kernel="boxcar", size=3, mirror=True)

        return ReturnTuple((ts, hr), ("index", "heart_rate"))

    hr_idx, hr = zip(*(map(hr_map, rpeaks)))

    # create time vectors
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = tuple(ts[r] for r in hr_idx)
    ts_tmpl = tuple(
        np.linspace(-0.2, 0.4, t.shape[1], endpoint=False) for t in templates
    )

    features = {
        "ts": ts,
        "filtered": filtered,
        "rpeak_det": rpeak_det,
        "rpeaks": rpeaks,
        "templates_ts": ts_tmpl,
        "templates": templates,
        "heart_rate_ts": ts_hr,
        "heart_rate": hr,
    }
    return features


def ensure_rpeaks_valid(rpeaks, length, filtered, sampling_rate, rpeak_det):
    ref = None
    for idx, rpeak in enumerate(rpeaks):
        if len(rpeak) < 2:
            if ref is None:
                ref = consensus([r for r in rpeaks if len(r) > 2], length=length)
            trial = {}

            try:
                # A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”, BIOSIGNALS 2012, pp. 49-54, 2012
                #  biosppy.signals.ecg.engzee_segmenter(signal=None, sampling_rate=1000.0, threshold=0.48)
                trial["engzee"] = engzee_segmenter(
                    filtered[idx], sampling_rate=sampling_rate
                )
            except Exception:
                pass

            try:
                # Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, BioMedical Engineering OnLine 2004, vol. 3:28, 2004
                #  biosppy.signals.ecg.christov_segmenter(signal=None, sampling_rate=1000.0)
                trial["christov"] = christov_segmenter(
                    filtered[idx], sampling_rate=sampling_rate
                )
            except Exception:
                pass

            try:
                # Gamboa
                #  biosppy.signals.ecg.gamboa_segmenter(signal=None, sampling_rate=1000.0, tol=0.002)
                trial["gamboa"] = gamboa_segmenter(
                    filtered[idx], sampling_rate=sampling_rate
                )
            except Exception:
                pass

            try:
                # Slope Sum Function (SSF)
                #  biosppy.signals.ecg.ssf_segmenter(signal=None, sampling_rate=1000.0, threshold=20, before=0.03, after=0.01)
                trial["ssf"] = ssf_segmenter(filtered[idx], sampling_rate=sampling_rate)
            except Exception:
                pass

            # Only keep the algorithms that could detect more than 2 beats
            trial = [
                (k, v["rpeaks"]) for (k, v) in trial.items() if len(v["rpeaks"]) > 2
            ]

            if trial:
                # pick the algorithm with best performance relative to the consensus hamilton

                choice = min(
                    trial,
                    key=lambda x: compare_segmentation(
                        reference=ref, test=x[1], sampling_rate=sampling_rate
                    )["performance"],
                )
            else:
                # no algorithm could detect peaks, default to the reference
                choice = (
                    "consensus",
                    ref,
                )
            rpeaks[idx] = choice[1]
            rpeak_det[idx] = choice[0]


def parse_header_data(header_data):
    # split header data and comments
    header_lines = []
    comment_lines = []
    for line in header_data:
        line = line.strip()
        if line.startswith("#"):
            comment_lines.append(line)
        elif line:
            ci = line.find("#")
            if ci > 0:
                header_lines.append(line[:ci])
                comment_lines.append(line[ci:])
            else:
                header_lines.append(line)

    # Get fields from record line
    record_fields = _header._parse_record_line(header_lines[0])
    signal_fields = _header._parse_signal_lines(header_lines[1:])

    # Set the comments field
    comments = [line.strip(" \t#") for line in comment_lines]
    raw_age, raw_sx, raw_dx, _rx, _hx, _sx = comments

    dx_grp = re.search(r"^Dx: (?P<dx>.*)$", raw_dx)
    target = [0.0] * len(LABELS)
    for dxi in dx_grp.group("dx").split(","):
        target[LABELS.index(dxi)] = 1.0

    age_grp = re.search(r"^Age: (?P<age>.*)$", raw_age)
    age = float(age_grp.group("age"))
    if math.isnan(age):
        age = -1.0

    sx_grp = re.search(r"^Sex: (?P<sx>.*)$", raw_sx)
    sex = [0.0, 0.0]
    sex[SEX.index(sx_grp.group("sx"))] = 1.0

    comment_fields = {"age": age, "target": target, "sex": sex}

    return {"record": record_fields, "signal": signal_fields, "comment": comment_fields}


def extract_record_features(
    data,
    headers,
    template_resample=60,
    include_fourier=False,
    window_function="hann",
    freq_cap=50,
    num_bins=100,
):
    header_data = parse_header_data(headers)
    ecg_features = extract_ecg_features(data)
    ts = ecg_features["ts"]

    features = OrderedDict()

    features["sex"] = np.array(header_data["comment"]["sex"])
    features["age"] = np.array(header_data["comment"]["age"])
    features["target"] = np.array(header_data["comment"]["target"])

    for idx in range(header_data["record"]["n_sig"]):
        sig_name = header_data["signal"]["sig_name"][idx]

        heart_rate = ecg_features["heart_rate"][idx]
        # heart_rate_ts = ecg_features["heart_rate_ts"][idx]
        rpeak_det = ecg_features["rpeak_det"][idx]
        rpeaks = ecg_features["rpeaks"][idx]
        templates = ecg_features["templates"][idx]
        # templates_ts = ecg_features["templates_ts"][idx]

        # r-peak features
        rpeak_det_alg = np.zeros(len(RPEAK_DET_ALG))
        rpeak_det_alg[RPEAK_DET_ALG.index(rpeak_det)] = 1.0
        features[f"l_{sig_name}_rp_det_alg"] = rpeak_det_alg

        # features[f"l_{sig_name}_rp_len"] = len(rpeaks)
        rel_rpeaks = np.diff(rpeaks)
        # this information is already used for heart rate... unnecessary?
        # features[f"l_{sig_name}_rp_len"] = len(rel_rpeaks)
        # features[f"l_{sig_name}_rp_max"] = np.max(rel_rpeaks)
        # features[f"l_{sig_name}_rp_min"] = np.min(rel_rpeaks)
        # features[f"l_{sig_name}_rp_median"] = np.median(rel_rpeaks)
        # features[f"l_{sig_name}_rp_mean"] = np.mean(rel_rpeaks)
        # features[f"l_{sig_name}_rp_std"] = np.std(rel_rpeaks)
        # features[f"l_{sig_name}_rp_var"] = np.var(rel_rpeaks)

        # Bin, Guangyu & Shao, Minggang & Guanghong, Bin & Huang, Jiao & Zheng, Dingchang & Wu, Shuicai. (2017).
        # Detection of Atrial Fibrillation Using Decision Tree Ensemble. 10.22489/CinC.2017.342-204.
        # Hand engineered RR interval features
        mrr = np.median(rel_rpeaks)
        rr_rule_1 = False  # 1.2 * RR2 < RR1 and 1.3 * RR2 < RR3
        rr_rule_2 = False  # |RR1 - RR2| < 0.3 * MRR and (RR1 < 0.8 * MRR or RR2 < 0.8 * MRR) and RR3 > 0.6 * (RR1 + RR2)
        rr_rule_3 = False  # |RR3 - RR2| < 0.3 * MRR and (RR2 < 0.8 * MRR or RR3 < 0.8 * MRR) and RR1 > 0.6 * (RR2 + RR3)
        rr_rule_4 = False  # RR2 > 1.5 * MRR and 1.5 * RR2 < 3 * MRR
        if len(rel_rpeaks) >= 3:
            for r_idx in range(len(rel_rpeaks) - 2):
                rr1 = rel_rpeaks[r_idx]
                rr2 = rel_rpeaks[r_idx + 1]
                rr3 = rel_rpeaks[r_idx + 2]

                rr_rule_1 = rr_rule_1 or ((1.2 * rr2 < rr1) and (1.3 * rr2 < rr3))
                rr_rule_2 = rr_rule_2 or (
                    (np.abs(rr1 - rr2) < (0.3 * mrr))
                    and ((rr1 < (0.8 * mrr)) or (rr2 < (0.8 * mrr)))
                    and (rr3 > (0.6 * (rr1 + rr2)))
                )
                rr_rule_3 = rr_rule_3 or (
                    (np.abs(rr3 - rr2) < (0.3 * mrr))
                    and ((rr2 < (0.8 * mrr)) or (rr3 < (0.8 * mrr)))
                    and (rr1 > (0.6 * (rr2 + rr3)))
                )
        rr_rules = np.zeros(8)
        if rr_rule_1:
            rr_rules[1] = 1.0
        else:
            rr_rules[0] = 1.0
        if rr_rule_2:
            rr_rules[3] = 1.0
        else:
            rr_rules[2] = 1.0
        if rr_rule_3:
            rr_rules[5] = 1.0
        else:
            rr_rules[4] = 1.0
        if rr_rule_4:
            rr_rules[7] = 1.0
        else:
            rr_rules[6] = 1.0

        features[f"l_{sig_name}_rr_int_rules"] = rr_rules

        # heart rate features
        features[f"l_{sig_name}_hr_len"] = np.array(len(heart_rate))
        heart_rate_data = np.zeros(6)
        # max heart rate
        heart_rate_data[0] = np.max(heart_rate)
        # min heart rate
        heart_rate_data[1] = np.min(heart_rate)
        # median heart rate
        heart_rate_data[2] = np.median(heart_rate)
        # mean heart rate
        heart_rate_data[3] = np.mean(heart_rate)
        # heart rate standard deviation
        heart_rate_data[4] = np.std(heart_rate)
        # heart rate variance
        heart_rate_data[5] = np.var(heart_rate)

        features[f"l_{sig_name}_hr"] = heart_rate_data

        if template_resample:
            templates = resample(templates, template_resample, axis=1)

        num_templates, num_samples = templates.shape
        templates_data = np.zeros((6, num_samples))

        for t_sample in range(num_samples):
            t_slice = templates[:, t_sample]
            # template max
            templates_data[0][t_sample] = np.max(t_slice)
            # template min
            templates_data[1][t_sample] = np.min(t_slice)
            # template median
            templates_data[2][t_sample] = np.median(t_slice)
            # template mean
            templates_data[3][t_sample] = np.mean(t_slice)
            # template standard deviation
            templates_data[4][t_sample] = np.std(t_slice)
            # template variance
            templates_data[4][t_sample] = np.var(t_slice)
        features[f"l_{sig_name}_templates"] = templates_data

        if include_fourier:

            # perform FFT on raw signal with window (default hann)
            raw_sig = data[idx, :]
            window_sig = windows.get_window(window_function, len(raw_sig)) * raw_sig
            x_mag = scipy.absolute(scipy.fft(window_sig))

            # only keep 0hz to frequency_cap range (default 50hz)
            end_idx = len(x_mag) // header_data["record"]["fs"] * freq_cap
            x_mag = x_mag[:end_idx]

            # Statistical mean over each frequency w.r.t bin index?
            bins = {}
            f = np.linspace(0, num_bins, len(x_mag))

            for (magnitude, frequency) in zip(x_mag, f):
                freq_idx = min(int(math.floor(frequency)), num_bins - 1)
                bin_vals = bins.get(freq_idx, [])
                bin_vals.append(magnitude)
                bins[freq_idx] = bin_vals

            x_fft_bins = np.zeros(num_bins)
            for (frequency, magnitudes) in bins.items():
                x_fft_bins[frequency] = np.mean(magnitudes)

            features[f"l_{sig_name}_fft"] = x_fft_bins

    return features
