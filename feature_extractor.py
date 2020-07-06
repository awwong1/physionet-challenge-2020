import neurokit2 as nk


def get_structured_lead_features(lead_signal, sampling_rate=500, lead_name=""):
    """From an ECG single lead, return feature values and corresponding dtype
    """

    ir_cols = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD',
               'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN',
               'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF', 'HRV_VLF',
               'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn',
               'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD2SD1', 'HRV_CSI', 'HRV_CVI',
               'HRV_CSI_Modified', 'HRV_SampEn']

    data = []
    dtype = []
    try:
        """
        nk.ecg_process
        --------------
        signals : DataFrame
            A DataFrame of the same length as the `ecg_signal` containing the following columns:
            - *"ECG_Raw"*: the raw signal.
            - *"ECG_Clean"*: the cleaned signal.
            - *"ECG_R_Peaks"*: the R-peaks marked as "1" in a list of zeros.
            - *"ECG_Rate"*: heart rate interpolated between R-peaks.
            - *"ECG_P_Peaks"*: the P-peaks marked as "1" in a list of zeros
            - *"ECG_Q_Peaks"*: the Q-peaks marked as "1" in a list of zeros .
            - *"ECG_S_Peaks"*: the S-peaks marked as "1" in a list of zeros.
            - *"ECG_T_Peaks"*: the T-peaks marked as "1" in a list of zeros.
            - *"ECG_P_Onsets"*: the P-onsets marked as "1" in a list of zeros.
            - *"ECG_P_Offsets"*: the P-offsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).
            - *"ECG_T_Onsets"*: the T-onsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).
            - *"ECG_T_Offsets"*: the T-offsets marked as "1" in a list of zeros.
            - *"ECG_R_Onsets"*: the R-onsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).
            - *"ECG_R_Offsets"*: the R-offsets marked as "1" in a list of zeros
                                (only when method in `ecg_delineate` is wavelet).
            - *"ECG_Phase_Atrial"*: cardiac phase, marked by "1" for systole
                and "0" for diastole.
            - *"ECG_Phase_Ventricular"*: cardiac phase, marked by "1" for systole and "0" for diastole.
            - *"ECG_Atrial_PhaseCompletion"*: cardiac phase (atrial) completion, expressed in percentage
                (from 0 to 1), representing the stage of the current cardiac phase.
            - *"ECG_Ventricular_PhaseCompletion"*: cardiac phase (ventricular) completion, expressed in
                percentage (from 0 to 1), representing the stage of the current cardiac phase.
        info : dict
            A dictionary containing the samples at which the R-peaks occur, accessible with the key
            "ECG_Peaks".
        """
        signals, info = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)

        # get interval related features
        """
        NOTE: Not all of these metrics are outputted. Taken from the Neurokit2 lib docstrings

        - *"ECG_Rate_Mean"*: the mean heart rate.
        Time Domain HRV metrics:
            - **HRV_RMSSD**: The square root of the mean of the sum of successive differences between
            adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
            therefore it is redundant to report correlations with both (Ciccone, 2017).
            - **HRV_MeanNN**: The mean of the RR intervals.
            - **HRV_SDNN**: The standard deviation of the RR intervals.
            - **HRV_SDSD**: The standard deviation of the successive differences between RR intervals.
            - **HRV_CVNN**: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR
            intervals (MeanNN).
            - **HRV_CVSD**: The root mean square of the sum of successive differences (RMSSD) divided by the
            mean of the RR intervals (MeanNN).
            - **HRV_MedianNN**: The median of the absolute values of the successive differences between RR intervals.
            - **HRV_MadNN**: The median absolute deviation of the RR intervals.
            - **HRV_HCVNN**: The median absolute deviation of the RR intervals (MadNN) divided by the median
            of the absolute differences of their successive differences (MedianNN).
            - **HRV_IQRNN**: The interquartile range (IQR) of the RR intervals.
            - **HRV_pNN50**: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
            - **HRV_pNN20**: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
            - **HRV_TINN**: A geometrical parameter of the HRV, or more specifically, the baseline width of
            the RR intervals distribution obtained by triangular interpolation, where the error of least
            squares determines the triangle. It is an approximation of the RR interval distribution.
            - **HRV_HTI**: The HRV triangular index, measuring the total number of RR intervals divded by the
            height of the RR intervals histogram.

        Frequency Domain HRV metrics:
            - **HRV_ULF**: The spectral power density pertaining to ultra low frequency band i.e., .0 to .0033 Hz
            by default.
            - **HRV_VLF**: The spectral power density pertaining to very low frequency band i.e., .0033 to .04 Hz
            by default.
            - **HRV_LF**: The spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
            - **HRV_HF**: The spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
            - **HRV_VHF**: The variability, or signal power, in very high frequency i.e., .4 to .5 Hz by default.
            - **HRV_LFn**: The normalized low frequency, obtained by dividing the low frequency power by
            the total power.
            - **HRV_HFn**: The normalized high frequency, obtained by dividing the low frequency power by
            the total power.
            - **HRV_LnHF**: The log transformed HF.

        Non-linear HRV metrics:
        - **Characteristics of the Poincaré Plot Geometry**:
            - **HRV_SD1**: SD1 is a measure of the spread of RR intervals on the Poincaré plot
            perpendicular to the line of identity. It is an index of short-term RR interval
            fluctuations, i.e., beat-to-beat variability. It is equivalent (although on another
            scale) to RMSSD, and therefore it is redundant to report correlations with both
            (Ciccone, 2017).
            - **HRV_SD2**: SD2 is a measure of the spread of RR intervals on the Poincaré plot along the
            line of identity. It is an index of long-term RR interval fluctuations.
            - **HRV_SD1SD2**: the ratio between short and long term fluctuations of the RR intervals
            (SD1 divided by SD2).
            - **HRV_S**: Area of ellipse described by SD1 and SD2 (``pi * SD1 * SD2``). It is
            proportional to *SD1SD2*.
            - **HRV_CSI**: The Cardiac Sympathetic Index (Toichi, 1997), calculated by dividing the
            longitudinal variability of the Poincaré plot (``4*SD2``) by its transverse variability (``4*SD1``).
            - **HRV_CVI**: The Cardiac Vagal Index (Toichi, 1997), equal to the logarithm of the product of
            longitudinal (``4*SD2``) and transverse variability (``4*SD1``).
            - **HRV_CSI_Modified**: The modified CSI (Jeppesen, 2014) obtained by dividing the square of
            the longitudinal variability by its transverse variability.
        - **Indices of Heart Rate Asymmetry (HRA), i.e., asymmetry of the Poincaré plot** (Yan, 2017):
            - **HRV_GI**: Guzik's Index, defined as the distance of points above line of identity (LI)
            to LI divided by the distance of all points in Poincaré plot to LI except those that
            are located on LI.
            - **HRV_SI**: Slope Index, defined as the phase angle of points above LI divided by the
            phase angle of all points in Poincaré plot except those that are located on LI.
            - **HRV_AI**: Area Index, defined as the cumulative area of the sectors corresponding to
            the points that are located above LI divided by the cumulative area of sectors
            corresponding to all points in the Poincaré plot except those that are located on LI.
            - **HRV_PI**: Porta's Index, defined as the number of points below LI divided by the total
            number of points in Poincaré plot except those that are located on LI.
            - **HRV_SD1d** and **HRV_SD1a**: short-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).
            - **HRV_C1d** and **HRV_C1a**: the contributions of heart rate decelerations and accelerations
            to short-term HRV, respectively (Piskorski,  2011).
            - **HRV_SD2d** and **HRV_SD2a**: long-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).
            - **HRV_C2d** and **HRV_C2a**: the contributions of heart rate decelerations and accelerations
            to long-term HRV, respectively (Piskorski,  2011).
            - **HRV_SDNNd** and **HRV_SDNNa**: total variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).
            - **HRV_Cd** and **HRV_Ca**: the total contributions of heart rate decelerations and
            accelerations to HRV.
        - **Indices of Heart Rate Fragmentation** (Costa, 2017):
            - **HRV_PIP**: Percentage of inflection points of the RR intervals series.
            - **HRV_IALS**: Inverse of the average length of the acceleration/deceleration segments.
            - **HRV_PSS**: Percentage of short segments.
            - **HRV_PAS**: IPercentage of NN intervals in alternation segments.
        - **Indices of Complexity**:
            - **HRV_ApEn**: The approximate entropy measure of HRV, calculated by `entropy_approximate()`.
            - **HRV_SampEn**: The sample entropy measure of HRV, calculated by `entropy_sample()`.
        """

        ir_df = nk.ecg_intervalrelated(signals, sampling_rate=sampling_rate)
        assert all(ir_df.columns == ir_cols), f"interval related feature column mismatch: {lead_name}"

        ir_data = ir_df.to_numpy()
        ir_elem_dtype = ir_data.dtype
        ir_dtype = []
        for k in ir_df.columns:
            ir_dtype.append((f"{lead_name}_{k}", ir_elem_dtype))

        # cast into numpy array, parse back out values and dtypes
        data = tuple(ir_data.tolist()[0])
        dtype = ir_dtype
    except Exception:
        # TODO: fill in the rest of the data with NaNs
        for k in ir_cols:
            key = f"{lead_name}_{k}"
            if key not in [d[0] for d in dtype]:
                data.append(float("nan"))
                dtype.append((key, "f8"))

    return data, dtype
