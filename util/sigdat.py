import re

from wfdb.io import _header, wrsamp, Record

from util.sigproc import parse_header_data


def write_wfdb_sample(data, header_data, write_dir=""):
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
        comments=comments
    )
    r.wrsamp(write_dir=write_dir)
