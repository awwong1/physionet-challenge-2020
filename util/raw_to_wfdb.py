import random

from wfdb.io import Record, _header


SIG_NAMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


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

    record_name = record_fields.get("record_name", None)
    if record_name is None:
        _id = random.randrange(1, 50000)
        record_name = f"X{_id}"
    else:
        for char in record_name:
            if not char.isalnum():
                _id = random.randrange(1, 50000)
                record_name = f"X{_id}"
                break

    file_names = signal_fields.get("file_name")
    if (
        not file_names
        or len(file_names) != n_sig
        or any([record_name not in fn for fn in file_names])
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
        # byte_offset = [24, ] * n_sig
        byte_offset = [0,] * n_sig

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
        record_name=record_name,  # record_name must only contain alphanumeric chars, not guaranteed
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
