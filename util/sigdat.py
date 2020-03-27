import math
import os
import re
from collections import OrderedDict
from subprocess import run
from tempfile import TemporaryDirectory

import numpy as np
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
    with TemporaryDirectory() as temp_dir:
        r.adc(inplace=True)
        r.wrsamp(write_dir=temp_dir)
        # Call ecgpuwave (requires it to be in path)
        r_pth = os.path.join(temp_dir, record_name)
        run(
            f"ecgpuwave -r {record_name} -a atr",
            cwd=temp_dir,
            shell=True,
            check=True,
        )
        ann = rdann(r_pth, "atr",)
        print(ann)

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
