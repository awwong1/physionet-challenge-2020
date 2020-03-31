import math
import os
import re
from collections import OrderedDict
from subprocess import run, DEVNULL
from tempfile import TemporaryDirectory

import numpy as np
import scipy
from scipy.signal import resample, windows
from wfdb.io import Record, _header, rdann

LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
SEX = ("Male", "Female")


def convert_to_wfdb_record(data, header_data):
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

    file_names = signal_fields.get("file_name")
    if file_names:
        for idx, fn in enumerate(file_names):
            file_names[idx] = fn.split(".")[0] + ".dat"

    r = Record(
        # p_signal=data.T,
        d_signal=data.T.astype(int),
        record_name=record_fields.get("record_name", "A000"),
        n_sig=record_fields.get("n_sig", 12),
        fs=record_fields.get("fs", 500),
        counter_freq=record_fields.get("counter_freq"),
        base_counter=record_fields.get("base_counter"),
        sig_len=record_fields.get("sig_len"),
        base_time=record_fields.get("base_time"),
        base_date=record_fields.get("base_date"),
        file_name=file_names,
        fmt=signal_fields.get("fmt"),
        samps_per_frame=signal_fields.get("samps_per_frame"),
        skew=signal_fields.get("skew"),
        byte_offset=signal_fields.get("byte_offset"),
        adc_gain=signal_fields.get("adc_gain"),
        baseline=signal_fields.get("baseline"),
        units=signal_fields.get("units"),
        adc_res=signal_fields.get("adc_res"),
        adc_zero=signal_fields.get("adc_zero"),
        init_value=signal_fields.get("init_value"),
        checksum=signal_fields.get("checksum"),
        block_size=signal_fields.get("block_size"),
        sig_name=signal_fields.get("sig_name"),
        comments=comments,
    )

    # convert d_signal to p_signal
    r.dac(inplace=True)
    return r


def extract_features(r, check_errors=True):
    """
    Given a wfdb.io.Record, extract relevant features for classifier by signal name
    """
    features = OrderedDict({})

    # Get all possible information from the comments
    # applies to the entire record
    _parse_comment_lines(r, features)

    # Get all relevant signal features from the record
    record_name = r.record_name
    seq_len, num_signals = r.p_signal.shape
    fs = r.fs

    if check_errors:
        assert (
            len(r.sig_name) == r.n_sig
        ), f"{record_name} len(sig_name) != n_sig, {r.sig_name} != {r.n_sig}"
        assert (
            num_signals == r.n_sig
        ), f"{record_name} p_signal.shape[1] != n_sig, {p_signal.shape}[1] != {r.n_sig}"
        assert fs >= 500, f"{record_name} sampling frequency fs < 500, got {fs}"

    # Run the ecgpuwave annotation program and get the relevant annotations
    signal_features = {}
    with TemporaryDirectory() as temp_dir:
        # convert the analogue p_signal into digital d_signal
        r.adc(inplace=True)
        r.wrsamp(write_dir=temp_dir)
        # convert the digital d_signal back to analogue p_signal
        r.dac(inplace=True)
        for sig_idx in range(num_signals):
            try:
                signal_features[r.sig_name[sig_idx]] = _extract_signal_features(
                    temp_dir, r, sig_idx
                )
            except Exception as e:
                # print(e)
                pass

    features["sig"] = signal_features

    return features


def _parse_comment_lines(r, features):
    # Get all possible information from the comments
    # applies to the entire record
    for comment in r.comments:
        dx_grp = re.search(r"^Dx: (?P<dx>.*)$", comment)
        if dx_grp:
            target = [0.0] * len(LABELS)
            for dxi in dx_grp.group("dx").split(","):
                target[LABELS.index(dxi)] = 1.0
            features["target"] = np.array(target)
            continue

        age_grp = re.search(r"^Age: (?P<age>.*)$", comment)
        if age_grp:
            age = float(age_grp.group("age"))
            if math.isnan(age):
                age = -1.0
            features["age"] = np.array([age,])
            continue

        sx_grp = re.search(r"^Sex: (?P<sx>.*)$", comment)
        if sx_grp:
            sex = [0.0, 0.0]  # Male, Female
            if sx_grp.group("sx").upper().startswith("M"):
                sex[0] = 1.0
            elif sx_grp.group("sx").upper().startswith("F"):
                sex[1] = 1.0
            else:
                # patient sex was not provided, leave as zeros for both
                pass
            features["sex"] = np.array(sex)
            continue


def _extract_signal_features(temp_dir, r, sig_idx):
    # Call ecgpuwave (requires it to be in path)
    r_pth = os.path.join(temp_dir, r.record_name)
    run(
        f"ecgpuwave -r {r.record_name} -a atr{sig_idx} -s {sig_idx}",
        cwd=temp_dir,
        shell=True,
        check=True,
        stdout=DEVNULL,
        stderr=DEVNULL
    )
    ann = rdann(r_pth, f"atr{sig_idx}",)
    signal_feature = {}
    # print(ann)

    idx_2_symb = list(zip(ann.sample, ann.symbol))

    # Get R-peak indicies
    r_idxs = list((idx) for (idx, symb) in idx_2_symb if symb == "N")
    # Get R-peak amplitudes
    r_amps = list(r.p_signal[:, sig_idx][idx] for idx in r_idxs)
    # Get R-peak amplitude statistics
    r_max = np.amax(r_amps)
    r_min = np.amin(r_amps)
    r_median = np.median(r_amps)
    r_mean = np.mean(r_amps)
    r_var = np.var(r_amps)
    r_std = np.std(r_amps)
    signal_feature["R-peak"] = np.array([r_max, r_min, r_median, r_mean, r_var, r_std])

    # Get P-wave peak indicies
    p_idxs = list((idx) for (idx, symb) in idx_2_symb if symb == "p")
    # Get P-wave peak ampitudes
    p_amps = list(r.p_signal[:, sig_idx][idx] for idx in p_idxs)
    # Get P-wave peak amplitude statistics
    p_max = np.amax(p_amps)
    p_min = np.amin(p_amps)
    p_median = np.median(p_amps)
    p_mean = np.mean(p_amps)
    p_var = np.var(p_amps)
    p_std = np.std(p_amps)
    signal_feature["P-peak"] = np.array([p_max, p_min, p_median, p_mean, p_var, p_std])

    # Get T-wave peak indicies
    t_idxs = list((idx) for (idx, symb) in idx_2_symb if symb == "t")
    # Get T-wave peak amplitudes
    t_amps = list(r.p_signal[:, sig_idx][idx] for idx in t_idxs)
    # Get T-wave peak amplitude statistics
    t_max = np.amax(t_amps)
    t_min = np.amin(t_amps)
    t_median = np.median(t_amps)
    t_mean = np.mean(t_amps)
    t_var = np.var(t_amps)
    t_std = np.std(t_amps)
    signal_feature["T-peak"] = np.array([t_max, t_min, t_median, t_mean, t_var, t_std])

    # Get heart rate as normal beats per minute (distance between R-peaks)
    hr = list(r.fs / interval * 60 for interval in np.diff(r_idxs))
    # Get heart rate statistics
    hr_max = np.amax(hr)
    hr_min = np.amin(hr)
    hr_median = np.median(hr)
    hr_mean = np.mean(hr)
    hr_var = np.var(hr)
    hr_std = np.std(hr)
    signal_feature["HR"] = np.array(
        [hr_max, hr_min, hr_median, hr_mean, hr_var, hr_std]
    )

    # Get the P-waveform durations, R-waveform durations, T-waveform durations
    p_wave_durations = []
    r_wave_durations = []
    t_wave_durations = []
    for idx in range(len(idx_2_symb) - 2):
        s_idx, s_symb = idx_2_symb[idx]
        _, m_symb = idx_2_symb[idx + 1]
        e_idx, e_symb = idx_2_symb[idx + 2]

        if s_symb == "(" and m_symb == "p" and e_symb == ")":
            p_wave_durations.append((e_idx - s_idx) / r.fs)
        elif s_symb == "(" and m_symb == "N" and e_symb == ")":
            r_wave_durations.append((e_idx - s_idx) / r.fs)
        elif s_symb == "(" and m_symb == "t" and e_symb == ")":
            t_wave_durations.append((e_idx - s_idx) / r.fs)
    p_wave_durations_max = np.amax(p_wave_durations)
    p_wave_durations_min = np.amin(p_wave_durations)
    p_wave_durations_median = np.median(p_wave_durations)
    p_wave_durations_mean = np.mean(p_wave_durations)
    p_wave_durations_var = np.var(p_wave_durations)
    p_wave_durations_std = np.std(p_wave_durations)
    signal_feature["P-wave"] = np.array(
        [
            p_wave_durations_max,
            p_wave_durations_min,
            p_wave_durations_median,
            p_wave_durations_mean,
            p_wave_durations_var,
            p_wave_durations_std,
        ]
    )

    r_wave_durations_max = np.amax(r_wave_durations)
    r_wave_durations_min = np.amin(r_wave_durations)
    r_wave_durations_median = np.median(r_wave_durations)
    r_wave_durations_mean = np.mean(r_wave_durations)
    r_wave_durations_var = np.var(r_wave_durations)
    r_wave_durations_std = np.std(r_wave_durations)
    signal_feature["R-wave"] = np.array(
        [
            r_wave_durations_max,
            r_wave_durations_min,
            r_wave_durations_median,
            r_wave_durations_mean,
            r_wave_durations_var,
            r_wave_durations_std,
        ]
    )
    t_wave_durations_max = np.amax(t_wave_durations)
    t_wave_durations_min = np.amin(t_wave_durations)
    t_wave_durations_median = np.median(t_wave_durations)
    t_wave_durations_mean = np.mean(t_wave_durations)
    t_wave_durations_var = np.var(t_wave_durations)
    t_wave_durations_std = np.std(t_wave_durations)
    signal_feature["T-wave"] = np.array(
        [
            t_wave_durations_max,
            t_wave_durations_min,
            t_wave_durations_median,
            t_wave_durations_mean,
            t_wave_durations_var,
            t_wave_durations_std,
        ]
    )

    # Get the PR-segment and ST-segment durations
    pr_segment_durations = []
    st_segment_durations = []
    for idx in range(len(idx_2_symb) - 3):
        _, s_symb = idx_2_symb[idx]
        s_idx, m_symb = idx_2_symb[idx + 1]
        e_idx, e_symb = idx_2_symb[idx + 2]
        _, f_symb = idx_2_symb[idx + 3]
        if s_symb == "p" and m_symb == ")" and e_symb == "(" and f_symb == "N":
            pr_segment_durations.append((e_idx - s_idx) / r.fs)
        elif s_symb == "N" and m_symb == ")" and e_symb == "(" and f_symb == "t":
            st_segment_durations.append((e_idx - s_idx) / r.fs)

    pr_segment_durations_max = np.amax(pr_segment_durations)
    pr_segment_durations_min = np.amin(pr_segment_durations)
    pr_segment_durations_median = np.median(pr_segment_durations)
    pr_segment_durations_mean = np.mean(pr_segment_durations)
    pr_segment_durations_var = np.var(pr_segment_durations)
    pr_segment_durations_std = np.std(pr_segment_durations)
    signal_feature["PR-segment"] = np.array(
        [
            pr_segment_durations_max,
            pr_segment_durations_min,
            pr_segment_durations_median,
            pr_segment_durations_mean,
            pr_segment_durations_var,
            pr_segment_durations_std,
        ]
    )

    st_segment_durations_max = np.amax(st_segment_durations)
    st_segment_durations_min = np.amin(st_segment_durations)
    st_segment_durations_median = np.median(st_segment_durations)
    st_segment_durations_mean = np.mean(st_segment_durations)
    st_segment_durations_var = np.var(st_segment_durations)
    st_segment_durations_std = np.std(st_segment_durations)
    signal_feature["ST-segment"] = np.array(
        [
            st_segment_durations_max,
            st_segment_durations_min,
            st_segment_durations_median,
            st_segment_durations_mean,
            st_segment_durations_var,
            st_segment_durations_std,
        ]
    )

    # Calculate the fast fourier transform on the signal, binned across 50 Hz
    raw_sig = r.p_signal[:, sig_idx]
    window_sig = windows.get_window("hann", len(raw_sig)) * raw_sig
    x_mag = scipy.absolute(scipy.fft(window_sig))

    # Only keep 0hz to 50hz
    end_idx = len(x_mag) // r.fs * 50
    x_mag = x_mag[:end_idx]

    bins = {}
    # 50 bins, one for each of the hz
    num_bins = 50
    for (magnitude, frequency) in zip(x_mag, np.linspace(0, num_bins, len(x_mag))):
        freq_idx = min(int(math.floor(frequency)), num_bins - 1)
        bin_vals = bins.get(freq_idx, [])
        bin_vals.append(magnitude)
        bins[freq_idx] = bin_vals

    x_fft_bins = np.zeros(num_bins)
    for (frequency, magnitudes) in bins.items():
        x_fft_bins[frequency] = np.mean(magnitudes)

    signal_feature["FFT"] = x_fft_bins
    return signal_feature
