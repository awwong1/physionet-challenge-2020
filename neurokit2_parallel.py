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

KEYS_INTERVALRELATED = [
    "ECG_Rate_Mean",
    "HRV_RMSSD",
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_MCVNN",
    "HRV_IQRNN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_TINN",
    "HRV_HTI",
    "HRV_ULF",
    "HRV_VLF",
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_CSI_Modified",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_C1d",
    "HRV_C1a",
    "HRV_SD1d",
    "HRV_SD1a",
    "HRV_C2d",
    "HRV_C2a",
    "HRV_SD2d",
    "HRV_SD2a",
    "HRV_Cd",
    "HRV_Ca",
    "HRV_SDNNd",
    "HRV_SDNNa",
    "HRV_ApEn",
    "HRV_SampEn",
]

KEYS_TSFRESH = [
    "abs_energy",
    "absolute_sum_of_changes",
    'agg_autocorrelation__f_agg_"mean"__maxlag_40',
    'agg_autocorrelation__f_agg_"median"__maxlag_40',
    'agg_autocorrelation__f_agg_"var"__maxlag_40',
    'agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
    'agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
    'agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
    'agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
    'agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"max"',
    'agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"mean"',
    'agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"',
    'agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"var"',
    'agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"',
    'agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
    'agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
    'agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
    'agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
    'agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
    'agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
    'agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
    'agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
    'agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"',
    'agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
    'agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"',
    'agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
    'agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
    'agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
    'agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
    'agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"',
    'agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
    'agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
    'agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
    'agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
    'agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
    'agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"',
    'agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"min"',
    'agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"var"',
    'agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
    'agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
    'agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
    'agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
    "approximate_entropy__m_2__r_0.1",
    "approximate_entropy__m_2__r_0.3",
    "approximate_entropy__m_2__r_0.5",
    "approximate_entropy__m_2__r_0.7",
    "approximate_entropy__m_2__r_0.9",
    "ar_coefficient__coeff_0__k_10",
    "ar_coefficient__coeff_10__k_10",
    "ar_coefficient__coeff_1__k_10",
    "ar_coefficient__coeff_2__k_10",
    "ar_coefficient__coeff_3__k_10",
    "ar_coefficient__coeff_4__k_10",
    "ar_coefficient__coeff_5__k_10",
    "ar_coefficient__coeff_6__k_10",
    "ar_coefficient__coeff_7__k_10",
    "ar_coefficient__coeff_8__k_10",
    "ar_coefficient__coeff_9__k_10",
    'augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
    'augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"',
    'augmented_dickey_fuller__attr_"usedlag"__autolag_"AIC"',
    "autocorrelation__lag_0",
    "autocorrelation__lag_1",
    "autocorrelation__lag_2",
    "autocorrelation__lag_3",
    "autocorrelation__lag_4",
    "autocorrelation__lag_5",
    "autocorrelation__lag_6",
    "autocorrelation__lag_7",
    "autocorrelation__lag_8",
    "autocorrelation__lag_9",
    "binned_entropy__max_bins_10",
    "c3__lag_1",
    "c3__lag_2",
    "c3__lag_3",
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.2__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
    'change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.8',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
    'change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
    'change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
    'change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
    'change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
    'change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
    'change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
    'change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
    'change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
    "cid_ce__normalize_False",
    "cid_ce__normalize_True",
    "count_above__t_0",
    "count_above_mean",
    "count_below__t_0",
    "count_below_mean",
    "cwt_coefficients__coeff_0__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_0__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_0__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_11__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_6__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)",
    "cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_0",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_1",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_2",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_3",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_4",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_5",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_6",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_7",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_8",
    "energy_ratio_by_chunks__num_segments_10__segment_focus_9",
    'fft_aggregated__aggtype_"centroid"',
    'fft_aggregated__aggtype_"kurtosis"',
    'fft_aggregated__aggtype_"skew"',
    'fft_aggregated__aggtype_"variance"',
    'fft_coefficient__attr_"abs"__coeff_0',
    'fft_coefficient__attr_"abs"__coeff_1',
    'fft_coefficient__attr_"abs"__coeff_10',
    'fft_coefficient__attr_"abs"__coeff_11',
    'fft_coefficient__attr_"abs"__coeff_12',
    'fft_coefficient__attr_"abs"__coeff_13',
    'fft_coefficient__attr_"abs"__coeff_14',
    'fft_coefficient__attr_"abs"__coeff_15',
    'fft_coefficient__attr_"abs"__coeff_16',
    'fft_coefficient__attr_"abs"__coeff_17',
    'fft_coefficient__attr_"abs"__coeff_18',
    'fft_coefficient__attr_"abs"__coeff_19',
    'fft_coefficient__attr_"abs"__coeff_2',
    'fft_coefficient__attr_"abs"__coeff_20',
    'fft_coefficient__attr_"abs"__coeff_21',
    'fft_coefficient__attr_"abs"__coeff_22',
    'fft_coefficient__attr_"abs"__coeff_23',
    'fft_coefficient__attr_"abs"__coeff_24',
    'fft_coefficient__attr_"abs"__coeff_25',
    'fft_coefficient__attr_"abs"__coeff_26',
    'fft_coefficient__attr_"abs"__coeff_27',
    'fft_coefficient__attr_"abs"__coeff_28',
    'fft_coefficient__attr_"abs"__coeff_29',
    'fft_coefficient__attr_"abs"__coeff_3',
    'fft_coefficient__attr_"abs"__coeff_30',
    'fft_coefficient__attr_"abs"__coeff_31',
    'fft_coefficient__attr_"abs"__coeff_32',
    'fft_coefficient__attr_"abs"__coeff_33',
    'fft_coefficient__attr_"abs"__coeff_34',
    'fft_coefficient__attr_"abs"__coeff_35',
    'fft_coefficient__attr_"abs"__coeff_36',
    'fft_coefficient__attr_"abs"__coeff_37',
    'fft_coefficient__attr_"abs"__coeff_38',
    'fft_coefficient__attr_"abs"__coeff_39',
    'fft_coefficient__attr_"abs"__coeff_4',
    'fft_coefficient__attr_"abs"__coeff_40',
    'fft_coefficient__attr_"abs"__coeff_41',
    'fft_coefficient__attr_"abs"__coeff_42',
    'fft_coefficient__attr_"abs"__coeff_43',
    'fft_coefficient__attr_"abs"__coeff_44',
    'fft_coefficient__attr_"abs"__coeff_45',
    'fft_coefficient__attr_"abs"__coeff_46',
    'fft_coefficient__attr_"abs"__coeff_47',
    'fft_coefficient__attr_"abs"__coeff_48',
    'fft_coefficient__attr_"abs"__coeff_49',
    'fft_coefficient__attr_"abs"__coeff_5',
    'fft_coefficient__attr_"abs"__coeff_50',
    'fft_coefficient__attr_"abs"__coeff_51',
    'fft_coefficient__attr_"abs"__coeff_52',
    'fft_coefficient__attr_"abs"__coeff_53',
    'fft_coefficient__attr_"abs"__coeff_54',
    'fft_coefficient__attr_"abs"__coeff_55',
    'fft_coefficient__attr_"abs"__coeff_56',
    'fft_coefficient__attr_"abs"__coeff_57',
    'fft_coefficient__attr_"abs"__coeff_58',
    'fft_coefficient__attr_"abs"__coeff_59',
    'fft_coefficient__attr_"abs"__coeff_6',
    'fft_coefficient__attr_"abs"__coeff_60',
    'fft_coefficient__attr_"abs"__coeff_61',
    'fft_coefficient__attr_"abs"__coeff_62',
    'fft_coefficient__attr_"abs"__coeff_63',
    'fft_coefficient__attr_"abs"__coeff_64',
    'fft_coefficient__attr_"abs"__coeff_65',
    'fft_coefficient__attr_"abs"__coeff_66',
    'fft_coefficient__attr_"abs"__coeff_67',
    'fft_coefficient__attr_"abs"__coeff_68',
    'fft_coefficient__attr_"abs"__coeff_69',
    'fft_coefficient__attr_"abs"__coeff_7',
    'fft_coefficient__attr_"abs"__coeff_70',
    'fft_coefficient__attr_"abs"__coeff_71',
    'fft_coefficient__attr_"abs"__coeff_72',
    'fft_coefficient__attr_"abs"__coeff_73',
    'fft_coefficient__attr_"abs"__coeff_74',
    'fft_coefficient__attr_"abs"__coeff_75',
    'fft_coefficient__attr_"abs"__coeff_76',
    'fft_coefficient__attr_"abs"__coeff_77',
    'fft_coefficient__attr_"abs"__coeff_78',
    'fft_coefficient__attr_"abs"__coeff_79',
    'fft_coefficient__attr_"abs"__coeff_8',
    'fft_coefficient__attr_"abs"__coeff_80',
    'fft_coefficient__attr_"abs"__coeff_81',
    'fft_coefficient__attr_"abs"__coeff_82',
    'fft_coefficient__attr_"abs"__coeff_83',
    'fft_coefficient__attr_"abs"__coeff_84',
    'fft_coefficient__attr_"abs"__coeff_85',
    'fft_coefficient__attr_"abs"__coeff_86',
    'fft_coefficient__attr_"abs"__coeff_87',
    'fft_coefficient__attr_"abs"__coeff_88',
    'fft_coefficient__attr_"abs"__coeff_89',
    'fft_coefficient__attr_"abs"__coeff_9',
    'fft_coefficient__attr_"abs"__coeff_90',
    'fft_coefficient__attr_"abs"__coeff_91',
    'fft_coefficient__attr_"abs"__coeff_92',
    'fft_coefficient__attr_"abs"__coeff_93',
    'fft_coefficient__attr_"abs"__coeff_94',
    'fft_coefficient__attr_"abs"__coeff_95',
    'fft_coefficient__attr_"abs"__coeff_96',
    'fft_coefficient__attr_"abs"__coeff_97',
    'fft_coefficient__attr_"abs"__coeff_98',
    'fft_coefficient__attr_"abs"__coeff_99',
    'fft_coefficient__attr_"angle"__coeff_0',
    'fft_coefficient__attr_"angle"__coeff_1',
    'fft_coefficient__attr_"angle"__coeff_10',
    'fft_coefficient__attr_"angle"__coeff_11',
    'fft_coefficient__attr_"angle"__coeff_12',
    'fft_coefficient__attr_"angle"__coeff_13',
    'fft_coefficient__attr_"angle"__coeff_14',
    'fft_coefficient__attr_"angle"__coeff_15',
    'fft_coefficient__attr_"angle"__coeff_16',
    'fft_coefficient__attr_"angle"__coeff_17',
    'fft_coefficient__attr_"angle"__coeff_18',
    'fft_coefficient__attr_"angle"__coeff_19',
    'fft_coefficient__attr_"angle"__coeff_2',
    'fft_coefficient__attr_"angle"__coeff_20',
    'fft_coefficient__attr_"angle"__coeff_21',
    'fft_coefficient__attr_"angle"__coeff_22',
    'fft_coefficient__attr_"angle"__coeff_23',
    'fft_coefficient__attr_"angle"__coeff_24',
    'fft_coefficient__attr_"angle"__coeff_25',
    'fft_coefficient__attr_"angle"__coeff_26',
    'fft_coefficient__attr_"angle"__coeff_27',
    'fft_coefficient__attr_"angle"__coeff_28',
    'fft_coefficient__attr_"angle"__coeff_29',
    'fft_coefficient__attr_"angle"__coeff_3',
    'fft_coefficient__attr_"angle"__coeff_30',
    'fft_coefficient__attr_"angle"__coeff_31',
    'fft_coefficient__attr_"angle"__coeff_32',
    'fft_coefficient__attr_"angle"__coeff_33',
    'fft_coefficient__attr_"angle"__coeff_34',
    'fft_coefficient__attr_"angle"__coeff_35',
    'fft_coefficient__attr_"angle"__coeff_36',
    'fft_coefficient__attr_"angle"__coeff_37',
    'fft_coefficient__attr_"angle"__coeff_38',
    'fft_coefficient__attr_"angle"__coeff_39',
    'fft_coefficient__attr_"angle"__coeff_4',
    'fft_coefficient__attr_"angle"__coeff_40',
    'fft_coefficient__attr_"angle"__coeff_41',
    'fft_coefficient__attr_"angle"__coeff_42',
    'fft_coefficient__attr_"angle"__coeff_43',
    'fft_coefficient__attr_"angle"__coeff_44',
    'fft_coefficient__attr_"angle"__coeff_45',
    'fft_coefficient__attr_"angle"__coeff_46',
    'fft_coefficient__attr_"angle"__coeff_47',
    'fft_coefficient__attr_"angle"__coeff_48',
    'fft_coefficient__attr_"angle"__coeff_49',
    'fft_coefficient__attr_"angle"__coeff_5',
    'fft_coefficient__attr_"angle"__coeff_50',
    'fft_coefficient__attr_"angle"__coeff_51',
    'fft_coefficient__attr_"angle"__coeff_52',
    'fft_coefficient__attr_"angle"__coeff_53',
    'fft_coefficient__attr_"angle"__coeff_54',
    'fft_coefficient__attr_"angle"__coeff_55',
    'fft_coefficient__attr_"angle"__coeff_56',
    'fft_coefficient__attr_"angle"__coeff_57',
    'fft_coefficient__attr_"angle"__coeff_58',
    'fft_coefficient__attr_"angle"__coeff_59',
    'fft_coefficient__attr_"angle"__coeff_6',
    'fft_coefficient__attr_"angle"__coeff_60',
    'fft_coefficient__attr_"angle"__coeff_61',
    'fft_coefficient__attr_"angle"__coeff_62',
    'fft_coefficient__attr_"angle"__coeff_63',
    'fft_coefficient__attr_"angle"__coeff_64',
    'fft_coefficient__attr_"angle"__coeff_65',
    'fft_coefficient__attr_"angle"__coeff_66',
    'fft_coefficient__attr_"angle"__coeff_67',
    'fft_coefficient__attr_"angle"__coeff_68',
    'fft_coefficient__attr_"angle"__coeff_69',
    'fft_coefficient__attr_"angle"__coeff_7',
    'fft_coefficient__attr_"angle"__coeff_70',
    'fft_coefficient__attr_"angle"__coeff_71',
    'fft_coefficient__attr_"angle"__coeff_72',
    'fft_coefficient__attr_"angle"__coeff_73',
    'fft_coefficient__attr_"angle"__coeff_74',
    'fft_coefficient__attr_"angle"__coeff_75',
    'fft_coefficient__attr_"angle"__coeff_76',
    'fft_coefficient__attr_"angle"__coeff_77',
    'fft_coefficient__attr_"angle"__coeff_78',
    'fft_coefficient__attr_"angle"__coeff_79',
    'fft_coefficient__attr_"angle"__coeff_8',
    'fft_coefficient__attr_"angle"__coeff_80',
    'fft_coefficient__attr_"angle"__coeff_81',
    'fft_coefficient__attr_"angle"__coeff_82',
    'fft_coefficient__attr_"angle"__coeff_83',
    'fft_coefficient__attr_"angle"__coeff_84',
    'fft_coefficient__attr_"angle"__coeff_85',
    'fft_coefficient__attr_"angle"__coeff_86',
    'fft_coefficient__attr_"angle"__coeff_87',
    'fft_coefficient__attr_"angle"__coeff_88',
    'fft_coefficient__attr_"angle"__coeff_89',
    'fft_coefficient__attr_"angle"__coeff_9',
    'fft_coefficient__attr_"angle"__coeff_90',
    'fft_coefficient__attr_"angle"__coeff_91',
    'fft_coefficient__attr_"angle"__coeff_92',
    'fft_coefficient__attr_"angle"__coeff_93',
    'fft_coefficient__attr_"angle"__coeff_94',
    'fft_coefficient__attr_"angle"__coeff_95',
    'fft_coefficient__attr_"angle"__coeff_96',
    'fft_coefficient__attr_"angle"__coeff_97',
    'fft_coefficient__attr_"angle"__coeff_98',
    'fft_coefficient__attr_"angle"__coeff_99',
    'fft_coefficient__attr_"imag"__coeff_0',
    'fft_coefficient__attr_"imag"__coeff_1',
    'fft_coefficient__attr_"imag"__coeff_10',
    'fft_coefficient__attr_"imag"__coeff_11',
    'fft_coefficient__attr_"imag"__coeff_12',
    'fft_coefficient__attr_"imag"__coeff_13',
    'fft_coefficient__attr_"imag"__coeff_14',
    'fft_coefficient__attr_"imag"__coeff_15',
    'fft_coefficient__attr_"imag"__coeff_16',
    'fft_coefficient__attr_"imag"__coeff_17',
    'fft_coefficient__attr_"imag"__coeff_18',
    'fft_coefficient__attr_"imag"__coeff_19',
    'fft_coefficient__attr_"imag"__coeff_2',
    'fft_coefficient__attr_"imag"__coeff_20',
    'fft_coefficient__attr_"imag"__coeff_21',
    'fft_coefficient__attr_"imag"__coeff_22',
    'fft_coefficient__attr_"imag"__coeff_23',
    'fft_coefficient__attr_"imag"__coeff_24',
    'fft_coefficient__attr_"imag"__coeff_25',
    'fft_coefficient__attr_"imag"__coeff_26',
    'fft_coefficient__attr_"imag"__coeff_27',
    'fft_coefficient__attr_"imag"__coeff_28',
    'fft_coefficient__attr_"imag"__coeff_29',
    'fft_coefficient__attr_"imag"__coeff_3',
    'fft_coefficient__attr_"imag"__coeff_30',
    'fft_coefficient__attr_"imag"__coeff_31',
    'fft_coefficient__attr_"imag"__coeff_32',
    'fft_coefficient__attr_"imag"__coeff_33',
    'fft_coefficient__attr_"imag"__coeff_34',
    'fft_coefficient__attr_"imag"__coeff_35',
    'fft_coefficient__attr_"imag"__coeff_36',
    'fft_coefficient__attr_"imag"__coeff_37',
    'fft_coefficient__attr_"imag"__coeff_38',
    'fft_coefficient__attr_"imag"__coeff_39',
    'fft_coefficient__attr_"imag"__coeff_4',
    'fft_coefficient__attr_"imag"__coeff_40',
    'fft_coefficient__attr_"imag"__coeff_41',
    'fft_coefficient__attr_"imag"__coeff_42',
    'fft_coefficient__attr_"imag"__coeff_43',
    'fft_coefficient__attr_"imag"__coeff_44',
    'fft_coefficient__attr_"imag"__coeff_45',
    'fft_coefficient__attr_"imag"__coeff_46',
    'fft_coefficient__attr_"imag"__coeff_47',
    'fft_coefficient__attr_"imag"__coeff_48',
    'fft_coefficient__attr_"imag"__coeff_49',
    'fft_coefficient__attr_"imag"__coeff_5',
    'fft_coefficient__attr_"imag"__coeff_50',
    'fft_coefficient__attr_"imag"__coeff_51',
    'fft_coefficient__attr_"imag"__coeff_52',
    'fft_coefficient__attr_"imag"__coeff_53',
    'fft_coefficient__attr_"imag"__coeff_54',
    'fft_coefficient__attr_"imag"__coeff_55',
    'fft_coefficient__attr_"imag"__coeff_56',
    'fft_coefficient__attr_"imag"__coeff_57',
    'fft_coefficient__attr_"imag"__coeff_58',
    'fft_coefficient__attr_"imag"__coeff_59',
    'fft_coefficient__attr_"imag"__coeff_6',
    'fft_coefficient__attr_"imag"__coeff_60',
    'fft_coefficient__attr_"imag"__coeff_61',
    'fft_coefficient__attr_"imag"__coeff_62',
    'fft_coefficient__attr_"imag"__coeff_63',
    'fft_coefficient__attr_"imag"__coeff_64',
    'fft_coefficient__attr_"imag"__coeff_65',
    'fft_coefficient__attr_"imag"__coeff_66',
    'fft_coefficient__attr_"imag"__coeff_67',
    'fft_coefficient__attr_"imag"__coeff_68',
    'fft_coefficient__attr_"imag"__coeff_69',
    'fft_coefficient__attr_"imag"__coeff_7',
    'fft_coefficient__attr_"imag"__coeff_70',
    'fft_coefficient__attr_"imag"__coeff_71',
    'fft_coefficient__attr_"imag"__coeff_72',
    'fft_coefficient__attr_"imag"__coeff_73',
    'fft_coefficient__attr_"imag"__coeff_74',
    'fft_coefficient__attr_"imag"__coeff_75',
    'fft_coefficient__attr_"imag"__coeff_76',
    'fft_coefficient__attr_"imag"__coeff_77',
    'fft_coefficient__attr_"imag"__coeff_78',
    'fft_coefficient__attr_"imag"__coeff_79',
    'fft_coefficient__attr_"imag"__coeff_8',
    'fft_coefficient__attr_"imag"__coeff_80',
    'fft_coefficient__attr_"imag"__coeff_81',
    'fft_coefficient__attr_"imag"__coeff_82',
    'fft_coefficient__attr_"imag"__coeff_83',
    'fft_coefficient__attr_"imag"__coeff_84',
    'fft_coefficient__attr_"imag"__coeff_85',
    'fft_coefficient__attr_"imag"__coeff_86',
    'fft_coefficient__attr_"imag"__coeff_87',
    'fft_coefficient__attr_"imag"__coeff_88',
    'fft_coefficient__attr_"imag"__coeff_89',
    'fft_coefficient__attr_"imag"__coeff_9',
    'fft_coefficient__attr_"imag"__coeff_90',
    'fft_coefficient__attr_"imag"__coeff_91',
    'fft_coefficient__attr_"imag"__coeff_92',
    'fft_coefficient__attr_"imag"__coeff_93',
    'fft_coefficient__attr_"imag"__coeff_94',
    'fft_coefficient__attr_"imag"__coeff_95',
    'fft_coefficient__attr_"imag"__coeff_96',
    'fft_coefficient__attr_"imag"__coeff_97',
    'fft_coefficient__attr_"imag"__coeff_98',
    'fft_coefficient__attr_"imag"__coeff_99',
    'fft_coefficient__attr_"real"__coeff_0',
    'fft_coefficient__attr_"real"__coeff_1',
    'fft_coefficient__attr_"real"__coeff_10',
    'fft_coefficient__attr_"real"__coeff_11',
    'fft_coefficient__attr_"real"__coeff_12',
    'fft_coefficient__attr_"real"__coeff_13',
    'fft_coefficient__attr_"real"__coeff_14',
    'fft_coefficient__attr_"real"__coeff_15',
    'fft_coefficient__attr_"real"__coeff_16',
    'fft_coefficient__attr_"real"__coeff_17',
    'fft_coefficient__attr_"real"__coeff_18',
    'fft_coefficient__attr_"real"__coeff_19',
    'fft_coefficient__attr_"real"__coeff_2',
    'fft_coefficient__attr_"real"__coeff_20',
    'fft_coefficient__attr_"real"__coeff_21',
    'fft_coefficient__attr_"real"__coeff_22',
    'fft_coefficient__attr_"real"__coeff_23',
    'fft_coefficient__attr_"real"__coeff_24',
    'fft_coefficient__attr_"real"__coeff_25',
    'fft_coefficient__attr_"real"__coeff_26',
    'fft_coefficient__attr_"real"__coeff_27',
    'fft_coefficient__attr_"real"__coeff_28',
    'fft_coefficient__attr_"real"__coeff_29',
    'fft_coefficient__attr_"real"__coeff_3',
    'fft_coefficient__attr_"real"__coeff_30',
    'fft_coefficient__attr_"real"__coeff_31',
    'fft_coefficient__attr_"real"__coeff_32',
    'fft_coefficient__attr_"real"__coeff_33',
    'fft_coefficient__attr_"real"__coeff_34',
    'fft_coefficient__attr_"real"__coeff_35',
    'fft_coefficient__attr_"real"__coeff_36',
    'fft_coefficient__attr_"real"__coeff_37',
    'fft_coefficient__attr_"real"__coeff_38',
    'fft_coefficient__attr_"real"__coeff_39',
    'fft_coefficient__attr_"real"__coeff_4',
    'fft_coefficient__attr_"real"__coeff_40',
    'fft_coefficient__attr_"real"__coeff_41',
    'fft_coefficient__attr_"real"__coeff_42',
    'fft_coefficient__attr_"real"__coeff_43',
    'fft_coefficient__attr_"real"__coeff_44',
    'fft_coefficient__attr_"real"__coeff_45',
    'fft_coefficient__attr_"real"__coeff_46',
    'fft_coefficient__attr_"real"__coeff_47',
    'fft_coefficient__attr_"real"__coeff_48',
    'fft_coefficient__attr_"real"__coeff_49',
    'fft_coefficient__attr_"real"__coeff_5',
    'fft_coefficient__attr_"real"__coeff_50',
    'fft_coefficient__attr_"real"__coeff_51',
    'fft_coefficient__attr_"real"__coeff_52',
    'fft_coefficient__attr_"real"__coeff_53',
    'fft_coefficient__attr_"real"__coeff_54',
    'fft_coefficient__attr_"real"__coeff_55',
    'fft_coefficient__attr_"real"__coeff_56',
    'fft_coefficient__attr_"real"__coeff_57',
    'fft_coefficient__attr_"real"__coeff_58',
    'fft_coefficient__attr_"real"__coeff_59',
    'fft_coefficient__attr_"real"__coeff_6',
    'fft_coefficient__attr_"real"__coeff_60',
    'fft_coefficient__attr_"real"__coeff_61',
    'fft_coefficient__attr_"real"__coeff_62',
    'fft_coefficient__attr_"real"__coeff_63',
    'fft_coefficient__attr_"real"__coeff_64',
    'fft_coefficient__attr_"real"__coeff_65',
    'fft_coefficient__attr_"real"__coeff_66',
    'fft_coefficient__attr_"real"__coeff_67',
    'fft_coefficient__attr_"real"__coeff_68',
    'fft_coefficient__attr_"real"__coeff_69',
    'fft_coefficient__attr_"real"__coeff_7',
    'fft_coefficient__attr_"real"__coeff_70',
    'fft_coefficient__attr_"real"__coeff_71',
    'fft_coefficient__attr_"real"__coeff_72',
    'fft_coefficient__attr_"real"__coeff_73',
    'fft_coefficient__attr_"real"__coeff_74',
    'fft_coefficient__attr_"real"__coeff_75',
    'fft_coefficient__attr_"real"__coeff_76',
    'fft_coefficient__attr_"real"__coeff_77',
    'fft_coefficient__attr_"real"__coeff_78',
    'fft_coefficient__attr_"real"__coeff_79',
    'fft_coefficient__attr_"real"__coeff_8',
    'fft_coefficient__attr_"real"__coeff_80',
    'fft_coefficient__attr_"real"__coeff_81',
    'fft_coefficient__attr_"real"__coeff_82',
    'fft_coefficient__attr_"real"__coeff_83',
    'fft_coefficient__attr_"real"__coeff_84',
    'fft_coefficient__attr_"real"__coeff_85',
    'fft_coefficient__attr_"real"__coeff_86',
    'fft_coefficient__attr_"real"__coeff_87',
    'fft_coefficient__attr_"real"__coeff_88',
    'fft_coefficient__attr_"real"__coeff_89',
    'fft_coefficient__attr_"real"__coeff_9',
    'fft_coefficient__attr_"real"__coeff_90',
    'fft_coefficient__attr_"real"__coeff_91',
    'fft_coefficient__attr_"real"__coeff_92',
    'fft_coefficient__attr_"real"__coeff_93',
    'fft_coefficient__attr_"real"__coeff_94',
    'fft_coefficient__attr_"real"__coeff_95',
    'fft_coefficient__attr_"real"__coeff_96',
    'fft_coefficient__attr_"real"__coeff_97',
    'fft_coefficient__attr_"real"__coeff_98',
    'fft_coefficient__attr_"real"__coeff_99',
    "first_location_of_maximum",
    "first_location_of_minimum",
    "friedrich_coefficients__coeff_0__m_3__r_30",
    "friedrich_coefficients__coeff_1__m_3__r_30",
    "friedrich_coefficients__coeff_2__m_3__r_30",
    "friedrich_coefficients__coeff_3__m_3__r_30",
    "has_duplicate",
    "has_duplicate_max",
    "has_duplicate_min",
    "index_mass_quantile__q_0.1",
    "index_mass_quantile__q_0.2",
    "index_mass_quantile__q_0.3",
    "index_mass_quantile__q_0.4",
    "index_mass_quantile__q_0.6",
    "index_mass_quantile__q_0.7",
    "index_mass_quantile__q_0.8",
    "index_mass_quantile__q_0.9",
    "kurtosis",
    "large_standard_deviation__r_0.05",
    "large_standard_deviation__r_0.1",
    "large_standard_deviation__r_0.15000000000000002",
    "large_standard_deviation__r_0.2",
    "large_standard_deviation__r_0.25",
    "large_standard_deviation__r_0.30000000000000004",
    "large_standard_deviation__r_0.35000000000000003",
    "large_standard_deviation__r_0.4",
    "large_standard_deviation__r_0.45",
    "large_standard_deviation__r_0.5",
    "large_standard_deviation__r_0.55",
    "large_standard_deviation__r_0.6000000000000001",
    "large_standard_deviation__r_0.65",
    "large_standard_deviation__r_0.7000000000000001",
    "large_standard_deviation__r_0.75",
    "large_standard_deviation__r_0.8",
    "large_standard_deviation__r_0.8500000000000001",
    "large_standard_deviation__r_0.9",
    "large_standard_deviation__r_0.9500000000000001",
    "last_location_of_maximum",
    "last_location_of_minimum",
    "length",
    'linear_trend__attr_"intercept"',
    'linear_trend__attr_"pvalue"',
    'linear_trend__attr_"rvalue"',
    'linear_trend__attr_"slope"',
    'linear_trend__attr_"stderr"',
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "max_langevin_fixed_point__m_3__r_30",
    "maximum",
    "mean",
    "mean_abs_change",
    "mean_change",
    "mean_second_derivative_central",
    "median",
    "minimum",
    "number_crossing_m__m_-1",
    "number_crossing_m__m_0",
    "number_crossing_m__m_1",
    "number_cwt_peaks__n_1",
    "number_cwt_peaks__n_5",
    "number_peaks__n_1",
    "number_peaks__n_10",
    "number_peaks__n_3",
    "number_peaks__n_5",
    "number_peaks__n_50",
    "partial_autocorrelation__lag_0",
    "partial_autocorrelation__lag_1",
    "partial_autocorrelation__lag_2",
    "partial_autocorrelation__lag_3",
    "partial_autocorrelation__lag_4",
    "partial_autocorrelation__lag_5",
    "partial_autocorrelation__lag_6",
    "partial_autocorrelation__lag_7",
    "partial_autocorrelation__lag_8",
    "partial_autocorrelation__lag_9",
    "percentage_of_reoccurring_datapoints_to_all_datapoints",
    "percentage_of_reoccurring_values_to_all_values",
    "quantile__q_0.1",
    "quantile__q_0.2",
    "quantile__q_0.3",
    "quantile__q_0.4",
    "quantile__q_0.6",
    "quantile__q_0.7",
    "quantile__q_0.8",
    "quantile__q_0.9",
    "range_count__max_0__min_1000000000000.0",
    "range_count__max_1000000000000.0__min_0",
    "range_count__max_1__min_-1",
    "ratio_beyond_r_sigma__r_0.5",
    "ratio_beyond_r_sigma__r_1",
    "ratio_beyond_r_sigma__r_1.5",
    "ratio_beyond_r_sigma__r_10",
    "ratio_beyond_r_sigma__r_2",
    "ratio_beyond_r_sigma__r_2.5",
    "ratio_beyond_r_sigma__r_3",
    "ratio_beyond_r_sigma__r_5",
    "ratio_beyond_r_sigma__r_6",
    "ratio_beyond_r_sigma__r_7",
    "ratio_value_number_to_time_series_length",
    "sample_entropy",
    "skewness",
    "spkt_welch_density__coeff_2",
    "spkt_welch_density__coeff_5",
    "spkt_welch_density__coeff_8",
    "standard_deviation",
    "sum_of_reoccurring_data_points",
    "sum_of_reoccurring_values",
    "sum_values",
    "symmetry_looking__r_0.0",
    "symmetry_looking__r_0.05",
    "symmetry_looking__r_0.1",
    "symmetry_looking__r_0.15000000000000002",
    "symmetry_looking__r_0.2",
    "symmetry_looking__r_0.25",
    "symmetry_looking__r_0.30000000000000004",
    "symmetry_looking__r_0.35000000000000003",
    "symmetry_looking__r_0.4",
    "symmetry_looking__r_0.45",
    "symmetry_looking__r_0.5",
    "symmetry_looking__r_0.55",
    "symmetry_looking__r_0.6000000000000001",
    "symmetry_looking__r_0.65",
    "symmetry_looking__r_0.7000000000000001",
    "symmetry_looking__r_0.75",
    "symmetry_looking__r_0.8",
    "symmetry_looking__r_0.8500000000000001",
    "symmetry_looking__r_0.9",
    "symmetry_looking__r_0.9500000000000001",
    "time_reversal_asymmetry_statistic__lag_1",
    "time_reversal_asymmetry_statistic__lag_2",
    "time_reversal_asymmetry_statistic__lag_3",
    "value_count__value_-1",
    "value_count__value_0",
    "value_count__value_1",
    "variance",
    "variance_larger_than_standard_deviation",
    "variation_coefficient",
]


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

    # rpeaks not guaranteed to be sorted in increasing order?
    rpeaks = np.sort(rpeaks)
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
        peaks = np.sort(peaks)
        period = np.sort(period)
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
    sig_len, num_leads = ecg_signal.shape
    quality = dict(
        map(
            functools.partial(_ecg_quality_partial, sampling_rate=sampling_rate),
            zip(ecg_signal.T, all_r_peaks, range(num_leads)),
        )
    )

    return quality


def _ecg_quality_partial(sig_info_idx, sampling_rate=500):
    signal, info, lead_idx = sig_info_idx
    rpeaks = info["ECG_R_Peaks"]
    if len(rpeaks) == 0:
        quality = np.full(len(signal), np.nan)
    else:
        try:
            quality = nk.ecg_quality(
                signal, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
            )
        except Exception:
            quality = np.full(len(signal), np.nan)
    return lead_idx, quality


def ecg_delineate(
    ecg_signal, all_r_peaks, sampling_rate=500, ecg_lead_names=ECG_LEAD_NAMES
):
    """somewhat parallelized version of nk.ecg_delinate,
    calculates P, Q, S, T peaks, P onsets, T offsets
    """
    delineate_info = map(
        functools.partial(_ecg_delineate_partial, sampling_rate=sampling_rate),
        zip(ecg_signal.T, all_r_peaks, ecg_lead_names),
    )
    delinate_dfs, delineate_infos = zip(*delineate_info)
    delinate_dfs = pd.concat(delinate_dfs)
    return delinate_dfs, delineate_infos


def _ecg_delineate_partial(sig_info_idx_name, sampling_rate=500):
    signal, info, lead_name = sig_info_idx_name
    sig_len = len(signal)
    try:
        ref = nk.ecg_delineate(
            ecg_cleaned=signal, rpeaks=info, sampling_rate=sampling_rate
        )
    except Exception:
        ref = (
            pd.DataFrame(
                {
                    "ECG_P_Peaks": [0.0,] * sig_len,
                    "ECG_Q_Peaks": [0.0,] * sig_len,
                    "ECG_S_Peaks": [0.0,] * sig_len,
                    "ECG_T_Peaks": [0.0,] * sig_len,
                    "ECG_P_Onsets": [0.0,] * sig_len,
                    "ECG_T_Offsets": [0.0,] * sig_len,
                }
            ),
            {
                "ECG_P_Peaks": [],
                "ECG_Q_Peaks": [],
                "ECG_S_Peaks": [],
                "ECG_T_Peaks": [],
                "ECG_P_Onsets": [],
                "ECG_T_Offsets": [],
            },
        )
    ref[0]["ECG_Sig_Name"] = lead_name
    return ref


def ecg_intervalrelated(proc_df, sampling_rate=500, ecg_lead_names=ECG_LEAD_NAMES):
    """multi-lead version of nk.ecg_intervalrelated
    """
    df_groupby = proc_df.groupby("ECG_Sig_Name")
    record_feats = []
    for ecg_lead_name in ecg_lead_names:
        try:
            lead_feats = nk.ecg_intervalrelated(
                df_groupby.get_group(ecg_lead_name), sampling_rate=sampling_rate
            )
        except Exception:
            lead_feats = pd.DataFrame.from_dict(
                dict((k, (np.nan,)) for k in KEYS_INTERVALRELATED)
            )
        finally:
            lead_feats["ECG_Sig_Name"] = ecg_lead_name
        record_feats.append(lead_feats)
    record_feats = pd.concat(record_feats)
    return record_feats


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


def get_intervalrelated_features(
    raw_signals, cleaned_signals, sampling_rate=500, ecg_lead_names=ECG_LEAD_NAMES
):
    # interval related features from parallel neurokit2
    sig_len, num_leads = cleaned_signals.shape

    df_sig_names = []
    for ln in ECG_LEAD_NAMES:
        df_sig_names += [ln,] * sig_len

    peaks_df, peaks_info = ecg_peaks(
        cleaned_signals, sampling_rate=sampling_rate, ecg_lead_names=ecg_lead_names
    )

    # non-df outputs...
    rate = signal_rate(peaks_info, sampling_rate=sampling_rate, desired_length=sig_len)
    quality = ecg_quality(cleaned_signals, peaks_info, sampling_rate=sampling_rate)

    rate_values = np.concatenate([rate[lead_idx] for lead_idx in range(num_leads)])
    quality_values = np.concatenate(
        [quality[lead_idx] for lead_idx in range(num_leads)]
    )

    proc_df = pd.DataFrame(
        {
            "ECG_Raw": raw_signals.flatten(order="F"),
            "ECG_Clean": cleaned_signals.flatten(order="F"),
            "ECG_Sig_Name": df_sig_names,
            "ECG_R_Peaks": peaks_df["ECG_R_Peaks"],
            "ECG_Rate": rate_values,
            "ECG_Quality": quality_values,
        }
    )
    ir_features = ecg_intervalrelated(proc_df, sampling_rate=sampling_rate)
    return ir_features, proc_df


def best_heartbeats_from_ecg_signal(
    proc_df, sampling_rate=500, ecg_lead_names=ECG_LEAD_NAMES
):
    """utility method for tsfresh feature extraction
    """
    best_heartbeats = []
    for lead_name in ecg_lead_names:
        try:
            lead_df = proc_df[proc_df["ECG_Sig_Name"] == lead_name]
            heartbeats = nk.ecg_segment(
                lead_df.rename(columns={"ECG_Clean": "Signal"}).drop(
                    columns=["ECG_Raw"]
                ),
                rpeaks=np.where(lead_df["ECG_R_Peaks"] > 0)[0],
                show=False,
            )

            best_idx = None
            best_quality = -1
            for k, v in heartbeats.items():
                if not all(np.isfinite(v["Signal"])):
                    continue
                hb_quality_stats = scipy.stats.describe(v["ECG_Quality"])
                if hb_quality_stats.mean > best_quality:
                    best_idx = k
                    best_quality = hb_quality_stats.mean

            best_heartbeat = heartbeats[best_idx]["Signal"]
            hb_num_samples = len(best_heartbeat)
            hb_duration = hb_num_samples / sampling_rate
            hb_times = np.linspace(0, hb_duration, hb_num_samples).tolist()

            hb_input_df = pd.DataFrame(
                {
                    "lead": [lead_name,] * hb_num_samples,
                    "time": hb_times,
                    "hb_sig": best_heartbeat.tolist(),
                }
            )
        except Exception:
            hb_num_samples = 10
            hb_times = np.linspace(0, 1.0, hb_num_samples).tolist()
            hb_input_df = pd.DataFrame(
                {
                    "lead": [lead_name,] * hb_num_samples,
                    "time": hb_times,
                    "hb_sig": [0.0,] * hb_num_samples,
                }
            )

        best_heartbeats.append(hb_input_df)
    best_heartbeats = pd.concat(best_heartbeats)
    return best_heartbeats


def signal_to_tsfresh_df(
    cleaned_signals,
    sampling_rate=500,
    ecg_lead_names=ECG_LEAD_NAMES,
    get_num_samples=2000,
    mod_fs=500,
):
    # convert sampling rate to mod_fs
    if sampling_rate != mod_fs:
        len_mod_fs = int(len(cleaned_signals) / sampling_rate * mod_fs)
        cleaned_signals = scipy.signal.resample(cleaned_signals, len_mod_fs)

    # drop 1 second from the start and ends of the cleaned signals
    cleaned_signals = cleaned_signals[mod_fs:-mod_fs]
    num_samples, num_leads = cleaned_signals.shape

    # if over get_num_samples, take middle
    if num_samples > get_num_samples:
        mid_point = int(num_samples / 2)
        cleaned_signals = cleaned_signals[
            mid_point - get_num_samples // 2 : mid_point + get_num_samples // 2  # noqa: E203
        ]
        num_samples, num_leads = cleaned_signals.shape

    duration = num_samples / mod_fs
    # convert to tsfresh compatible dataframe
    df_sig_names = []
    for ln in ecg_lead_names:
        df_sig_names += [ln,] * num_samples

    times = np.linspace(0, duration, num_samples).tolist()
    input_df = pd.DataFrame(
        {
            "lead": df_sig_names,
            "time": times * num_leads,
            "sig": cleaned_signals.flatten(order="F"),
        }
    )

    return input_df
