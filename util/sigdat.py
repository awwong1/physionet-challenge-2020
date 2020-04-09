import math
import os
import re
from collections import Counter, OrderedDict
from functools import partial
from multiprocessing import Pool
from shutil import copy
from subprocess import DEVNULL, run
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import scipy.signal as ss
import wfdb
from scipy.signal import resample, windows
from wfdb.io import Record, _header, rdann
from wfdb import processing

LABELS = ("AF", "I-AVB", "LBBB", "Normal", "PAC", "PVC", "RBBB", "STD", "STE")
SEX = ("Male", "Female")
SIG_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ANN_SYMBS = ("(", ")", "p", "N", "t")


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

    # The Physionet 2020 header file cannot be trusted!
    # Massage data if very wrong
    n_sig, sig_len = data.shape  # do not trust provided n_sig or sig_len

    record_name = record_fields.get("record_name", "A9999")
    for char in record_name:
        if not char.isalnum():
            record_name = "A9999"
            break

    file_names = signal_fields.get("file_name")
    if (
        not file_names
        or len(file_names) != n_sig
        or any([record_name not in file_name for file_name in file_names])
    ):
        file_names = [f"{record_name}.dat",] * n_sig

    fmt = signal_fields.get("fmt")
    if not fmt or len(fmt) != n_sig:
        fmt = ["16",] * n_sig

    samps_per_frame = signal_fields.get("samps_per_frame")
    if not samps_per_frame or len(samps_per_frame) != n_sig:
        samps_per_frame = [1,] * n_sig

    skew = signal_fields.get("skew")
    if not skew or len(skew) != n_sig:
        skew = [None,] * n_sig

    byte_offset = signal_fields.get("byte_offset")
    if not byte_offset or len(byte_offset) != n_sig:
        byte_offset = [24,] * n_sig

    adc_gain = signal_fields.get("adc_gain")
    if not adc_gain or len(adc_gain) != n_sig:
        adc_gain = [1000.0,] * n_sig

    baseline = signal_fields.get("baseline")
    if not baseline or len(baseline) != n_sig:
        baseline = [0,] * n_sig

    units = signal_fields.get("units")
    if not units or len(units) != n_sig:
        units = ["mV",] * n_sig

    adc_res = signal_fields.get("adc_res")
    if not adc_res or len(adc_res) != n_sig:
        adc_res = [16,] * n_sig

    adc_zero = signal_fields.get("adc_zero")
    if not adc_zero or len(adc_zero) != n_sig:
        adc_zero = [0,] * n_sig

    block_size = signal_fields.get("block_size")
    if not block_size or len(block_size) != n_sig:
        block_size = [0,] * n_sig

    init_value = [int(x) for x in data[:, 0].tolist()]

    sig_name = signal_fields.get("sig_name")
    if not sig_name or len(sig_name) != n_sig:
        sig_name = SIG_NAMES[:n_sig]

    checksum = signal_fields.get("checksum")
    if not checksum or len(checksum) != n_sig:
        checksum = [0,] * n_sig

    r = Record(
        # p_signal=data.T,
        d_signal=data.T.astype(int),
        record_name="entry",  # record_name, # record_name must only contain alphanumeric chars, not guaranteed
        n_sig=n_sig,  # record_fields.get("n_sig", 12),
        fs=record_fields.get("fs", 500),
        counter_freq=record_fields.get("counter_freq"),
        base_counter=record_fields.get("base_counter"),
        sig_len=sig_len,  # record_fields.get("sig_len"),
        base_time=record_fields.get("base_time"),
        base_date=record_fields.get("base_date"),
        file_name=file_names,
        fmt=fmt,  # signal_fields.get("fmt"),
        samps_per_frame=samps_per_frame,  # signal_fields.get("samps_per_frame"),
        skew=skew,  # signal_fields.get("skew"),
        byte_offset=byte_offset,  # signal_fields.get("byte_offset"),
        adc_gain=adc_gain,  # signal_fields.get("adc_gain"),
        baseline=baseline,  # signal_fields.get("baseline"),
        units=units,  # signal_fields.get("units"),
        adc_res=adc_res,  # signal_fields.get("adc_res"),
        adc_zero=adc_zero,  # signal_fields.get("adc_zero"),
        init_value=init_value,  # signal_fields.get("init_value"),
        block_size=block_size,  # signal_fields.get("block_size"),
        sig_name=sig_name,  # signal_fields.get("sig_name"),
        checksum=checksum,  # signal_fields.get("checksum"),
        comments=comments,
    )

    # convert d_signal to p_signal
    r.dac(inplace=True)
    return r


def _parse_comment_lines(comments=[]):
    """Given a list of string comments, extract Physionet2020 relevant features.
    """
    target = [0.0] * len(LABELS)
    sex = [0.0, 0.0]  # Male, Female

    comment_features = OrderedDict(
        {"target": np.array(target), "age": np.array([-1,]), "sex": np.array(sex)}
    )
    if not comments:
        return comment_features

    for comment in comments:
        dx_grp = re.search(r"^Dx: (?P<dx>.*)$", comment)
        if dx_grp:
            for dxi in dx_grp.group("dx").split(","):
                target[LABELS.index(dxi)] = 1.0
            comment_features["target"] = np.array(target)
            continue

        age_grp = re.search(r"^Age: (?P<age>.*)$", comment)
        if age_grp:
            age = float(age_grp.group("age"))
            if math.isnan(age):
                age = -1.0
            comment_features["age"] = np.array([age,])
            continue

        sx_grp = re.search(r"^Sex: (?P<sx>.*)$", comment)
        if sx_grp:
            if sx_grp.group("sx").upper().startswith("M"):
                sex[0] = 1.0
            elif sx_grp.group("sx").upper().startswith("F"):
                sex[1] = 1.0
            else:
                # patient sex was not provided, leave as zeros for both
                pass
            comment_features["sex"] = np.array(sex)
            continue

    return comment_features


def _run_ecgpuwave(sig_idx, record_name=None, temp_dir=None, write_dir=""):
    """
    Call the ecgpuwave fortran binary.

    Parameters
    ----------
    sig_idx : int
        index of the current wfdb.Record signal to analyze
    record_name : str
        record name to run ecgpuwave on
    temp_dir : str
        path to directory containing the corresponding wfdb.Record
    write_dir : str (optional)
        path to directory to store wfdb.Annotation output files

    Returns
    -------
    sig_idx : int
        signal index ecgpuwave ran on
    ann : wfdb.Annotation, None
        ecgpuwave annotation results, or None if an error occurred
    """
    if write_dir:
        # check to see if the record has already been processed
        c_pth = os.path.join(write_dir, record_name)
        try:
            return sig_idx, wfdb.rdann(c_pth, f"atr{sig_idx}")
        except Exception:
            # nope, fallback to calling ecgpuwave
            pass

    r_pth = os.path.join(temp_dir, record_name)

    ann = None
    try:
        run(
            f"ecgpuwave -r {record_name} -a atr{sig_idx} -s {sig_idx}",
            cwd=temp_dir,
            shell=True,
            check=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        _ann = wfdb.rdann(r_pth, f"atr{sig_idx}")
        if len(_ann.sample):
            # if the write_dir is set, copy the annotation file there
            if write_dir:
                copy(f"{r_pth}.atr{sig_idx}", f"{c_pth}")
            ann = _ann
    except Exception:
        # ecgpuwave failed to annotate the signal
        pass
    return sig_idx, ann


def _get_descriptive_stats(a):
    """Get a numpy vector representing the scipy descriptive stats of the given iterable
    """
    desc_out = np.array([-1,] * 6)
    try:
        desc = scipy.stats.describe(a, axis=None)
        desc_out = np.array(
            [
                desc.minmax[0],
                desc.minmax[1],
                desc.mean,
                desc.variance,
                desc.skewness,
                desc.kurtosis,
            ]
        )
    except Exception:
        pass
    return desc_out


def _calculate_durations(idx_2_symb, sampling_rate=500):
    """Calculate all relevant ECG signal durations using pass through the symbol indicies

    Parameters
    ----------
    idx_2_symb : Iterable
        Contains (idx, symb) tuples, where symb in ["(", ")", "p", "N", "t"] and
        idx is an integer >= 0 < seq_len. Nested/overlapping waveforms are not supported.
    sampling_rate : int (optional)
        Frequency of idx_2_symb sample indicies (Hz), defaults to 500
    Returns
    -------
    all_durations : dict
        Keys are tuples representing measurement, distance is always between parenthesis `(` and `)`, symbols are reference
        Values are lists of times (reported as seconds according to sampling_rate)
    """
    all_durations = {}

    prev_lb_idx = None  # When did the last waveform start
    cur_lb_idx = None  # When does the current waveform start
    prev_rb_idx = None  # When did the last waveform end
    cur_rb_idx = None  # Where does the current waveform end
    cur_wf_symbs = None  # What symbols are in the current waveform
    prev_wf_symbs = None  # What symbols were in the last waveform

    for (idx, symb) in idx_2_symb:
        if symb == "(":
            prev_lb_idx = cur_lb_idx
            cur_lb_idx = idx
            cur_wf_symbs = []
        elif symb == ")":
            # Waveform ended
            prev_rb_idx = cur_rb_idx
            cur_rb_idx = idx

            # calculate durations
            waveform_duration = (cur_rb_idx - cur_lb_idx) / sampling_rate
            duration_key = "".join(cur_wf_symbs)
            if duration_key == "":
                duration_key = " "
            durations = all_durations.get(("(", duration_key, ")"), [])
            durations.append(waveform_duration)
            all_durations[("(", duration_key, ")")] = durations

            # calculate segments
            if prev_wf_symbs:
                prev_duration_key = "".join(prev_wf_symbs)
                if prev_duration_key == "":
                    prev_duration_key = " "

                # lb_interval: `(, KEY, ), (, KEY` # from ( to (
                lb_interval = (cur_lb_idx - prev_lb_idx) / sampling_rate
                lb_intervals = all_durations.get(
                    ("(", prev_duration_key, "(", duration_key), []
                )
                lb_intervals.append(lb_interval)
                all_durations[
                    ("(", prev_duration_key, "(", duration_key)
                ] = lb_intervals

                # rb_interval: `KEY, ), (, KEY, )` # from ) to )
                rb_interval = (cur_rb_idx - prev_rb_idx) / sampling_rate
                rb_intervals = all_durations.get(
                    (prev_duration_key, ")", duration_key, ")"), []
                )
                rb_intervals.append(rb_interval)
                all_durations[
                    (prev_duration_key, ")", duration_key, ")")
                ] = rb_intervals

                # segment: `KEY, ), (, KEY` # from ) to (
                segment = (cur_lb_idx - prev_rb_idx) / sampling_rate
                segments = all_durations.get(
                    (prev_duration_key, ")", "(", duration_key), []
                )
                segments.append(segment)
                all_durations[(prev_duration_key, ")", "(", duration_key)] = segments

            prev_wf_symbs = cur_wf_symbs
        else:
            # Define current waveform type
            cur_wf_symbs.append(symb)

    return all_durations


def extract_features(r, ann_dir=None):
    """
    Given a wfdb.Record, extract relevant features for classifier by signal name.

    Parameters
    ----------
    r : wfdb.Record
        Required attributes: comments, record_name, p_signal, fs
    ann_dir : str, optional
        If provided, write extracted ecgpuwave annotations here
    """
    features = OrderedDict(_parse_comment_lines(r.comments))
    features["debug"] = {}
    _seq_len, num_signals = r.p_signal.shape
    record_name = r.record_name
    sampling_rate = r.fs

    # Inspired by BioSPPy, perform the same signal filtering step on 12 leads
    # =======================================================================
    # https://github.com/PIA-Group/BioSPPy/blob/212c3dcbdb1ec43b70ba7199deb5eb22bcb78fd0/biosppy/signals/ecg.py#L71
    order = int(0.3 * sampling_rate)
    if order % 2 == 0:
        order += 1
    sig_filter = ss.firwin(
        numtaps=order, cutoff=[3, 45], pass_zero=False, fs=sampling_rate
    )
    r.p_signal = ss.filtfilt(b=sig_filter, a=np.array([1,]), x=r.p_signal, axis=0)

    # Use ECGPUWAVE annotation program to get signal annotations
    # ==========================================================
    signal_annotations = {}  # sig_idx (0-11), ann (wfdb.Annotation, None)
    with TemporaryDirectory() as temp_dir:
        # convert the analogue p_signal into digital d_signal and back
        r.adc(inplace=True)
        r.wrsamp(write_dir=temp_dir)
        r.dac(inplace=True)

        worker_fn = partial(
            _run_ecgpuwave,
            record_name=record_name,
            temp_dir=temp_dir,
            write_dir=ann_dir,
        )
        signals = list(range(num_signals))

        try:
            assert not os.getppid(), "parent process exists, cannot use pool"
            with Pool(len(os.sched_getaffinity(0))) as p:
                signal_annotations = dict(p.imap(worker_fn, signals,))
        except AssertionError:
            # single process approach
            signal_annotations = dict([worker_fn(signal) for signal in signals])

    # Find all of the lead indicies that threw an ECGPUWAVE error
    failed_ecgpuwave = [k for k, v in signal_annotations.items() if v is None]
    for fe in failed_ecgpuwave:
        signal_annotations.pop(fe)

    # Iterate through the annotations to determine effective fallback annotation
    # ==========================================================================
    # A successful annotation must have at least 2 p, N, and t symbols
    # Compare with each other, picking the one with the highest true positive count
    sig_ann_idxs = list(signal_annotations.keys())
    sig_ann_tp_count = {k: 0 for k in sig_ann_idxs}
    sig_ann_candidates = []

    for i0, k0 in enumerate(sig_ann_idxs):
        ann0 = signal_annotations[k0]
        for i1, k1 in enumerate(sig_ann_idxs[i0 + 1 :]):
            ann1 = signal_annotations[k1]
            for ref_symb in ANN_SYMBS:
                ref_idxs0 = np.array(
                    [
                        idx
                        for (idx, symb) in zip(ann0.sample, ann0.symbol)
                        if symb == ref_symb
                    ]
                )
                ref_idxs1 = np.array(
                    [
                        idx
                        for (idx, symb) in zip(ann1.sample, ann1.symbol)
                        if symb == ref_symb
                    ]
                )
                try:
                    c = processing.compare_annotations(
                        ref_idxs0, ref_idxs1, int(0.02 * r.fs)
                    )
                    sig_ann_tp_count[k0] += c.tp
                    sig_ann_tp_count[k1] += c.tp
                except:
                    pass

    for sig_idx, ann in signal_annotations.items():
        symb_counter = Counter(ann.symbol)
        candidate = True
        for ref_symb in ANN_SYMBS:
            if symb_counter.get(ref_symb, 0) < 2:
                candidate = False
                break
        if candidate:
            sig_ann_candidates.append(sig_idx)

    # WARNING: if this throws an Exception, not a single lead in the wfdb.Record is useable
    fallback_ann_idx = max(sig_ann_candidates, key=(lambda k: sig_ann_tp_count[k]))

    # Calculate basic statistical descriptors for each lead
    # =====================================================
    features["sig"] = OrderedDict()
    for sig_idx, lead_name in enumerate(SIG_NAMES):
        signal_feature = {}
        ann = signal_annotations.get(sig_idx, signal_annotations[fallback_ann_idx])
        idx_2_symb = list(zip(ann.sample, ann.symbol))

        # Get P-wave peak indicies, amplitudes, stats
        p_idxs = [(idx) for (idx, symb) in idx_2_symb if symb == "p"]
        p_amps = [r.p_signal[:, sig_idx][idx] for idx in p_idxs]
        signal_feature["P-peak"] = _get_descriptive_stats(p_amps)

        # Get R-wave peak indicies, amplitudes, stats
        r_idxs = [(idx) for (idx, symb) in idx_2_symb if symb == "N"]
        r_amps = [r.p_signal[:, sig_idx][idx] for idx in r_idxs]
        signal_feature["R-peak"] = _get_descriptive_stats(r_amps)

        # Get T-wave peak indicies, amplitudes, stats
        t_idxs = [(idx) for (idx, symb) in idx_2_symb if symb == "t"]
        t_amps = [r.p_signal[:, sig_idx][idx] for idx in t_idxs]
        signal_feature["T-peak"] = _get_descriptive_stats(t_amps)

        # Get heart rate as normal beats per minute (distance between R-peaks)
        hr = [sampling_rate / interval * 60 for interval in np.diff(r_idxs)]
        signal_feature["HR"] = _get_descriptive_stats(hr)

        # Calculate the P-wave, R-wave, T-wave duration, PR & ST segment/interval
        raw_durations = _calculate_durations(idx_2_symb, sampling_rate=sampling_rate)
        # Get durations for P-wave, R-wave, T-wave TT-wave
        signal_feature["P-wave"] = _get_descriptive_stats(
            raw_durations.pop(("(", "p", ")"), [])
        )
        signal_feature["R-wave"] = _get_descriptive_stats(
            raw_durations.pop(("(", "N", ")"), [])
        )
        signal_feature["T-wave"] = _get_descriptive_stats(
            raw_durations.pop(("(", "t", ")"), [])
        )
        signal_feature["TT-wave"] = _get_descriptive_stats(
            raw_durations.pop(("(", "tt", ")"), [])
        )
        # Get durations for PR & ST segments and intervals, treating 'tt' as 't'
        signal_feature["PR-interval"] = _get_descriptive_stats(
            raw_durations.pop(("(", "p", "(", "N"), [])
        )
        signal_feature["PR-segment"] = _get_descriptive_stats(
            raw_durations.pop(("p", ")", "(", "N"), [])
        )
        st_intervals = raw_durations.pop(("(", "N", "(", "t"), []) + raw_durations.pop(
            ("(", "N", "(", "tt"), []
        )
        signal_feature["ST-interval"] = _get_descriptive_stats(st_intervals)
        st_segments = raw_durations.pop(("N", ")", "(", "t"), []) + raw_durations.pop(
            ("N", ")", "(", "tt"), []
        )
        signal_feature["ST-segment"] = _get_descriptive_stats(st_segments)

        # Get the unique symbols, check for weird edgecases
        unique_symbols = {key for keys in raw_durations.keys() for key in keys}
        supported_symbols = {"(", ")", "p", "N", "t", "tt"}
        unsupported_symbols = unique_symbols - supported_symbols
        features["debug"]["unsupported_symbols"] = unsupported_symbols

        # Set the signal feature into the record features dictionary
        features["sig"][lead_name] = signal_feature

    return features


def _extract_signal_features(sig_idx, temp_dir="", r=None):
    try:
        # Call ecgpuwave (requires it to be in path)
        r_pth = os.path.join(temp_dir, r.record_name)
        run(
            f"ecgpuwave -r {r.record_name} -a atr{sig_idx} -s {sig_idx}",
            cwd=temp_dir,
            shell=True,
            check=True,
            stdout=DEVNULL,
            stderr=DEVNULL,
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
        signal_feature["R-peak"] = np.array(
            [r_max, r_min, r_median, r_mean, r_var, r_std]
        )

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
        signal_feature["P-peak"] = np.array(
            [p_max, p_min, p_median, p_mean, p_var, p_std]
        )

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
        signal_feature["T-peak"] = np.array(
            [t_max, t_min, t_median, t_mean, t_var, t_std]
        )

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
        return r.sig_name[sig_idx], signal_feature
    except Exception:
        return None, None
