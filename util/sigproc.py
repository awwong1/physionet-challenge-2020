"""Functions for manual feature extraction of ECG 12 lead signals.
"""
from functools import partial

import numpy as np
from biosppy.signals.ecg import (
    christov_segmenter,
    correct_rpeaks,
    extract_heartbeats,
    hamilton_segmenter,
    engzee_segmenter,
    gamboa_segmenter,
    ssf_segmenter,
)
from biosppy.signals.tools import filter_signal, get_heart_rate


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
    #
    # some bad files:
    # A0115.mat
    # A0718.mat

    sampling_rate = float(sampling_rate)
    _num_leads, length = signal.shape

    # filter signal
    order = int(0.3 * sampling_rate)
    filtered = tuple(
        filter_signal(
            signal=sig,
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=[3, 45],
            sampling_rate=sampling_rate,
        )["signal"]
        for sig in signal
    )

    # segment
    rpeaks = list(
        hamilton_segmenter(signal=sig, sampling_rate=sampling_rate)["rpeaks"]
        for sig in filtered
    )
    rpeak_det = ["hamilton"] * len(rpeaks)

    # SANITY CHECK! If a a lead returns <2 peaks, it's an error!
    # try a different segmentation algorithm?
    for idx, rpeak in enumerate(rpeaks):
        if len(rpeak) < 2:
            trial = {
                # A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”, BIOSIGNALS 2012, pp. 49-54, 2012
                #  biosppy.signals.ecg.engzee_segmenter(signal=None, sampling_rate=1000.0, threshold=0.48)
                "engzee": engzee_segmenter(filtered[idx], sampling_rate=sampling_rate),
                # Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, BioMedical Engineering OnLine 2004, vol. 3:28, 2004
                #  biosppy.signals.ecg.christov_segmenter(signal=None, sampling_rate=1000.0)
                "christov": christov_segmenter(
                    filtered[idx], sampling_rate=sampling_rate
                ),
                # Gamboa
                #  biosppy.signals.ecg.gamboa_segmenter(signal=None, sampling_rate=1000.0, tol=0.002)
                "gamboa": gamboa_segmenter(filtered[idx], sampling_rate=sampling_rate),
                # Slope Sum Function (SSF)
                #  biosppy.signals.ecg.ssf_segmenter(signal=None, sampling_rate=1000.0, threshold=20, before=0.03, after=0.01)
                "ssf": ssf_segmenter(filtered[idx], sampling_rate=sampling_rate),
            }

            # pick the algorithm that returned a length closest to the others
            other_lens = [len(x) for x in rpeaks if len(x) >= 2]
            avg_other_lens = sum(other_lens) / len(other_lens)

            choice = min(
                [(k, v) for (k, v) in trial.items() if len(v["rpeaks"]) > 2],
                key=lambda x: abs(len(x[1]["rpeaks"]) - avg_other_lens),
            )
            rpeaks[idx] = choice[1]["rpeaks"]
            rpeak_det[idx] = choice[0]

    # correct R-peak locations
    rpeaks = tuple(
        correct_rpeaks(signal=sig, rpeaks=rpeak, sampling_rate=sampling_rate, tol=0.05)[
            "rpeaks"
        ]
        for (sig, rpeak) in zip(filtered, rpeaks)
    )

    # extract templates
    templates, rpeaks = zip(
        *(
            extract_heartbeats(
                signal=sig,
                rpeaks=rpeak,
                sampling_rate=sampling_rate,
                before=0.2,
                after=0.4,
            )
            for (sig, rpeak) in zip(filtered, rpeaks)
        )
    )

    # compute heart rate over time
    hr_idx, hr = zip(
        *(
            get_heart_rate(
                beats=rpeak, sampling_rate=sampling_rate, smooth=True, size=3
            )
            for rpeak in rpeaks
        )
    )

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
