import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
import os
import re

# Define the base dataset path
base_path = 'D:\\Machine learning project\\ptb-diagnostic-ecg-database-1.0.0'  # Replace with your dataset path
output_path = 'D:\Machine learning project\ecg_final.csv'  # Output file

# Create an empty DataFrame for storing features
features_df = pd.DataFrame()

for patient_folder in sorted(os.listdir(base_path)):
    patient_path = os.path.join(base_path, patient_folder)

    if os.path.isdir(patient_path):
        # Identify .dat and .hea files
        dat_file = []
        hea_file = []
        xyz_files = []
        for file in os.listdir(patient_path):
            if file.endswith('.dat'):
                dat_file.append(os.path.join(patient_path, file))
                #print(f'dat_file : {dat_file}')
            elif file.endswith('.hea'):
                hea_file.append(os.path.join(patient_path, file))
                #print(f'hea_file : {hea_file}')
            elif file.endswith('.xyz'):
                xyz_files.append(os.path.join(patient_path, file))
                #print(f'xyz_files : {xyz_files}')

        for i in range(len(dat_file)):
            if dat_file and hea_file:   
                try:
                    if hea_file[i]:
                        hea_content = open(hea_file[i], 'r').read()
                        with open(hea_file[i], 'r') as file:
                            first_line = file.readline().strip()
                        # Regular expression patterns to extract the required attributes
                        patterns = {
                            # Extract only the first part (ID) before any space
                            "age": r"# age:\s*(\d+)",
                            "sex": r"# sex:\s*(\w+)",
                            "Reason for admission": r"# Reason for admission:\s*(.*)",
                            "Acute infarction (localization)": r"# Acute infarction \(localization\):\s*(.*)",
                            "Smoker": r"# Smoker:\s*(\w+)",
                        }

                        # Extract the values based on the patterns
                        extracted_data = {}
                        extracted_data["Pat_ID"] = patient_folder
                        extracted_data["ID"] = first_line.split()[0]
                        for key, pattern in patterns.items():
                            match = re.search(pattern, hea_content)
                            if match:
                                extracted_data[key] = match.group(1)
                        
                            else:
                                extracted_data[key] = "NA"

                        #features_df = pd.concat([features_df, pd.DataFrame(extracted_data, index=[0])], ignore_index=False)

                    record = wfdb.rdrecord(dat_file[i][:-4])  # Strip .dat extension
                    print(dat_file[i][:-4])
                    sampling_rate = record.fs
                    leads = record.sig_name  # List of leads
                    # Assuming you have your ECG data in a NumPy array called 'ecg_signal'
                    Lead_val = []
                    for lead_index, lead_name in enumerate(leads[:3]):
                        ecg_signal = record.p_signal[:, lead_index]  # Choose the first lead (index 0)

                        # Get the sampling rate 
                        sampling_rate = record.fs

                        # Preprocess ECG signal
                        signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

                        # Example data
                        # If ECG_R_Peaks is a binary array
                        ECG_R_Peaks = np.array(signals['ECG_R_Peaks']) if 'ECG_R_Peaks' in signals else 0
                        fs = 1000  # Sampling rate (Hz)

                        # Step 1: Extract R-peak indices (if binary array)
                        R_peak_indices = np.where(ECG_R_Peaks == 1)[0] if 'ECG_R_Peaks' in signals else 0

                        # Step 2: Convert indices to timestamps (seconds)
                        R_peak_timestamps = R_peak_indices / fs

                        # Step 3: Calculate RR intervals (time difference between consecutive R-peaks)
                        RR_intervals = np.diff(R_peak_timestamps) if 'ECG_R_Peaks' in signals else 0

                        # Step 1: Extract indices of P onsets and R peaks (if binary arrays)
                        ECG_P_Onset = np.array(signals['ECG_P_Onsets'])

                        P_onset_indices = np.where(ECG_P_Onset == 1)[0]

                        # Step 2: Convert indices to timestamps (seconds)
                        P_onset_timestamps = P_onset_indices / fs
                        R_peak_timestamps = R_peak_indices / fs

                        # Step 3: Match each P onset to the nearest succeeding R peak
                        PR_intervals = []
                        for t_p in P_onset_timestamps:
                            # Find the closest succeeding R peak
                            succeeding_R_peaks = R_peak_timestamps[R_peak_timestamps > t_p]
                            if len(succeeding_R_peaks) > 0:
                                t_r = succeeding_R_peaks[0]  # Get the first (nearest) succeeding R peak
                                PR_intervals.append(t_r - t_p)

                        # Convert to numpy array for further processing
                        PR_intervals = np.array(PR_intervals)

                        ECG_Q_Peaks = np.array(signals['ECG_Q_Peaks'])
                        ECG_T_Offsets = np.array(signals['ECG_T_Offsets'])

                        # Step 1: Extract indices of Q peaks and T offsets (if binary arrays)
                        Q_peak_indices = np.where(ECG_Q_Peaks == 1)[0]
                        T_offset_indices = np.where(ECG_T_Offsets == 1)[0]

                        # Step 2: Convert indices to timestamps (seconds)
                        Q_peak_timestamps = Q_peak_indices / fs
                        T_offset_timestamps = T_offset_indices / fs

                        # Step 3: Match each Q peak to the nearest succeeding T offset
                        QT_intervals = []
                        for t_q in Q_peak_timestamps:
                            # Find the closest succeeding T offset
                            succeeding_T_offsets = T_offset_timestamps[T_offset_timestamps > t_q]
                            if len(succeeding_T_offsets) > 0:
                                t_t_offset = succeeding_T_offsets[0]  # Get the first (nearest) succeeding T offset
                                QT_intervals.append(t_t_offset - t_q)

                        # Convert to numpy array for further processing
                        QT_intervals = np.array(QT_intervals)

                        ECG_S_Peaks = np.array(signals['ECG_S_Peaks'])

                        # Assuming you have the R-peaks detected
                        R_peak_indices = np.where(signals['ECG_R_Peaks'] == 1)[0] if 'ECG_R_Peaks' in signals else 0 # Extract indices of R-peaks

                        # Compute HRV metrics using the detected R-peaks
                        hrv_metrics = nk.hrv_time(R_peak_indices, sampling_rate=1000)


                        ECG_P_peaks = signals['ECG_P_Peaks']
                        pp_intervals = np.diff(ECG_P_peaks)  # Convert to seconds

                        s_peaks = signals['ECG_S_Peaks']
                        t_onsets = signals['ECG_T_Onsets']

                        # Calculate ST segment durations
                        st_segments = np.array((t_onsets - s_peaks))/1000  # Convert to seconds

                        # Extract the onsets and offsets of R-peaks
                        r_onsets = signals['ECG_R_Onsets'].dropna().values
                        r_offsets = signals['ECG_R_Offsets'].dropna().values

                        # Ensure matching lengths
                        if len(r_onsets) != len(r_offsets):
                            raise ValueError("Mismatched lengths between R onsets and offsets.")

                        # Calculate QRS intervals (difference between offsets and onsets)
                        qrs_intervals = (r_offsets - r_onsets) / fs
                        features = {
                            f"{lead_name}_Mean_ECG_Rate": np.nanmean(signals['ECG_Rate']),
                            f"{lead_name}_Mean_ECG_Quality": 0 if 'ECG_Quality' not in signals else np.nanmean(np.array(signals['ECG_Quality'])),
                            f"{lead_name}_Coeff_R_peaks": 0 if 'ECG_R_Peaks' not in signals else np.nanstd(ECG_R_Peaks) / np.nanmean(ECG_R_Peaks),
                            #f"{lead_name}_Coeff_P_peaks": 0 if np.isnan(np.nanstd(ECG_P_peaks) / np.nanmean(ECG_P_peaks)) else np.nanstd(ECG_P_peaks) / np.nanmean(ECG_P_peaks),
                            #f"{lead_name}_Coeff_Q_peaks": 0 if np.isnan(np.nanstd(ECG_Q_Peaks) / np.nanmean(ECG_Q_Peaks)) else np.nanstd(ECG_Q_Peaks) / np.nanmean(ECG_Q_Peaks),
                            #f"{lead_name}_Coeff_S_peaks": 0 if np.isnan(np.nanstd(ECG_S_Peaks) / np.nanmean(ECG_S_Peaks)) else np.nanstd(ECG_S_Peaks) / np.nanmean(ECG_S_Peaks),
                            #f"{lead_name}_Coeff_T_peaks": 0 if np.any(np.isnan(signals['ECG_T_Peaks'])) else np.nanstd(signals['ECG_T_Peaks']) / np.nanmean(signals['ECG_T_Peaks']),
                            f"{lead_name}_Mean_ECG_Phase_Atrial": np.nanmean(signals['ECG_Phase_Atrial']),
                            f"{lead_name}_Mean_ECG_Phase_Completion_Atrial": np.nanmean(signals['ECG_Phase_Completion_Atrial']),
                            f"{lead_name}_Mean_ECG_Phase_Ventricular": np.nanmean(signals['ECG_Phase_Ventricular']),
                            f"{lead_name}_Mean_ECG_Phase_Completion_Ventricular": np.nanmean(signals['ECG_Phase_Completion_Ventricular']),
                            f"{lead_name}_Coeff_RR_intervals": 0 if np.isnan(np.nanstd(RR_intervals) / np.nanmean(RR_intervals)) else np.nanstd(RR_intervals) / np.nanmean(RR_intervals),
                            f"{lead_name}_Coeff_PR_intervals": 0 if np.isnan(np.nanstd(PR_intervals) / np.nanmean(PR_intervals)) else np.nanstd(PR_intervals) / np.nanmean(PR_intervals),
                            f"{lead_name}_Coeff_QT_intervals": 0 if np.isnan(np.mean(QT_intervals)) else np.nanstd(QT_intervals) / np.nanmean(QT_intervals),
                            f"{lead_name}_std_pp_intervals": 0 if np.isnan(np.nanstd(pp_intervals)) else np.nanstd(pp_intervals),
                            f"{lead_name}_std_QRS_Interval": 0 if np.isnan(np.nanstd(qrs_intervals)) else np.nanstd(qrs_intervals),
                            f"{lead_name}_HRV_SDNN": 0 if np.isnan(hrv_metrics['HRV_SDNN'].values[0]) else hrv_metrics['HRV_SDNN'].values[0],
                            f"{lead_name}_HRV_RMSSD": 0 if np.isnan(hrv_metrics['HRV_RMSSD'].values[0]) else hrv_metrics['HRV_RMSSD'].values[0],
                            f"{lead_name}_HRV_pNN20": 0 if np.isnan(hrv_metrics['HRV_pNN20'].values[0]) else hrv_metrics['HRV_pNN20'].values[0],
                        }

                        Lead_val.append(features)
                        #features_df = pd.concat([features_df, pd.DataFrame(features, index=[0])], ignore_index=False)
                    featurenet = extracted_data|Lead_val[0]|Lead_val[1]|Lead_val[2]
                    if xyz_files:
                        xyz_signals = []
                        record = wfdb.rdrecord(xyz_files[i][:-4])
                        leads = record.sig_name
                        
                        for i in range(1,len(leads[-3:])+1):
                            xyz_signal = record.p_signal[:, -i]
                            xyz_signals.append(xyz_signal)

                        # Compute magnitude vector if all three axes are present
                        if len(xyz_signals) == 3:
                            vx, vy, vz = xyz_signals
                            min_length = min(len(vx), len(vy), len(vz))
                            vx, vy, vz = vx[:min_length], vy[:min_length], vz[:min_length]
                            magnitude = np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
                            sampling_rate = 1000
                            # Preprocess ECG signal
                        signals, info = nk.ecg_process(magnitude, sampling_rate=sampling_rate)

                        # Example data
                        # If ECG_R_Peaks is a binary array
                        ECG_R_Peaks = np.array(signals['ECG_R_Peaks']) if 'ECG_R_Peaks' in signals else 0
                        fs = 1000  # Sampling rate (Hz)

                        # Step 1: Extract R-peak indices (if binary array)
                        R_peak_indices = np.where(ECG_R_Peaks == 1)[0] if 'ECG_R_Peaks' in signals else 0

                        # Step 2: Convert indices to timestamps (seconds)
                        R_peak_timestamps = R_peak_indices / fs

                        # Step 3: Calculate RR intervals (time difference between consecutive R-peaks)
                        RR_intervals = np.diff(R_peak_timestamps) if 'ECG_R_Peaks' in signals else 0

                        # Step 1: Extract indices of P onsets and R peaks (if binary arrays)
                        ECG_P_Onset = np.array(signals['ECG_P_Onsets'])

                        P_onset_indices = np.where(ECG_P_Onset == 1)[0]

                        # Step 2: Convert indices to timestamps (seconds)
                        P_onset_timestamps = P_onset_indices / fs
                        R_peak_timestamps = R_peak_indices / fs

                        # Step 3: Match each P onset to the nearest succeeding R peak
                        PR_intervals = []
                        for t_p in P_onset_timestamps:
                            # Find the closest succeeding R peak
                            succeeding_R_peaks = R_peak_timestamps[R_peak_timestamps > t_p]
                            if len(succeeding_R_peaks) > 0:
                                t_r = succeeding_R_peaks[0]  # Get the first (nearest) succeeding R peak
                                PR_intervals.append(t_r - t_p)

                        # Convert to numpy array for further processing
                        PR_intervals = np.array(PR_intervals)

                        ECG_Q_Peaks = np.array(signals['ECG_Q_Peaks'])
                        

                        ECG_T_Offsets = np.array(signals['ECG_T_Offsets'])

                        # Step 1: Extract indices of Q peaks and T offsets (if binary arrays)
                        Q_peak_indices = np.where(ECG_Q_Peaks == 1)[0]
                        T_offset_indices = np.where(ECG_T_Offsets == 1)[0]

                        # Step 2: Convert indices to timestamps (seconds)
                        Q_peak_timestamps = Q_peak_indices / fs
                        T_offset_timestamps = T_offset_indices / fs

                        # Step 3: Match each Q peak to the nearest succeeding T offset
                        QT_intervals = []
                        for t_q in Q_peak_timestamps:
                            # Find the closest succeeding T offset
                            succeeding_T_offsets = T_offset_timestamps[T_offset_timestamps > t_q]
                            if len(succeeding_T_offsets) > 0:
                                t_t_offset = succeeding_T_offsets[0]  # Get the first (nearest) succeeding T offset
                                QT_intervals.append(t_t_offset - t_q)

                        # Convert to numpy array for further processing
                        QT_intervals = np.array(QT_intervals)

                        ECG_S_Peaks = np.array(signals['ECG_S_Peaks'])

                        # Assuming you have the R-peaks detected
                        R_peak_indices = np.where(signals['ECG_R_Peaks'] == 1)[0]  # Extract indices of R-peaks

                        # Compute HRV metrics using the detected R-peaks
                        hrv_metrics = nk.hrv_time(R_peak_indices, sampling_rate=1000)


                        ECG_P_peaks = signals['ECG_P_Peaks']
                        pp_intervals = np.diff(ECG_P_peaks)  # Convert to seconds

                        s_peaks = signals['ECG_S_Peaks']
                        t_onsets = signals['ECG_T_Onsets']

                        # Calculate ST segment durations
                        st_segments = np.array((t_onsets - s_peaks))/1000  # Convert to seconds

                        # Extract the onsets and offsets of R-peaks
                        r_onsets = signals['ECG_R_Onsets'].dropna().values
                        r_offsets = signals['ECG_R_Offsets'].dropna().values

                        # Ensure matching lengths
                        if len(r_onsets) != len(r_offsets):
                            raise ValueError("Mismatched lengths between R onsets and offsets.")

                        # Calculate QRS intervals (difference between offsets and onsets)
                        qrs_intervals = (r_offsets - r_onsets) / fs

                        features = {
                            "0_Mean_ECG_Rate": np.nanmean(signals['ECG_Rate']) if 'ECG_Rate' in signals else 0,
                            "0_Mean_ECG_Quality": np.nanmean(np.array(signals['ECG_Quality'])) if 'ECG_Quality' in signals else 0,
                            "0_Coeff_R_peaks": 0 if 'ECG_R_Peaks' not in signals or np.isnan(np.nanmean(ECG_R_Peaks)) else np.nanstd(ECG_R_Peaks) / np.nanmean(ECG_R_Peaks),
                            #"0_Coeff_P_peaks": 0 if np.isnan(np.nanmean(ECG_P_peaks)) else np.nanstd(ECG_P_peaks) / np.nanmean(ECG_P_peaks),
                            #"0_Coeff_Q_peaks": 0 if np.isnan(np.nanmean(ECG_Q_Peaks)) else np.nanstd(ECG_Q_Peaks) / np.nanmean(ECG_Q_Peaks),
                            #"0_Coeff_S_peaks": 0 if np.isnan(np.nanmean(ECG_S_Peaks)) else np.nanstd(ECG_S_Peaks) / np.nanmean(ECG_S_Peaks),
                            #"0_Coeff_T_peaks": 0 if 'ECG_T_Peaks' not in signals or np.any(np.isnan(signals['ECG_T_Peaks'])) else np.nanstd(signals['ECG_T_Peaks']) / np.nanmean(signals['ECG_T_Peaks']),
                            "0_Mean_ECG_Phase_Atrial": np.nanmean(signals['ECG_Phase_Atrial']) if 'ECG_Phase_Atrial' in signals else 0,
                            "0_Mean_ECG_Phase_Completion_Atrial": np.nanmean(signals['ECG_Phase_Completion_Atrial']) if 'ECG_Phase_Completion_Atrial' in signals else 0,
                            "0_Mean_ECG_Phase_Ventricular": np.nanmean(signals['ECG_Phase_Ventricular']) if 'ECG_Phase_Ventricular' in signals else 0,
                            "0_Mean_ECG_Phase_Completion_Ventricular": np.nanmean(signals['ECG_Phase_Completion_Ventricular']) if 'ECG_Phase_Completion_Ventricular' in signals else 0,
                            "0_Coeff_RR_intervals": 0 if np.isnan(np.nanmean(RR_intervals)) else np.nanstd(RR_intervals) / np.nanmean(RR_intervals),
                            "0_Coeff_PR_intervals": 0 if np.isnan(np.nanmean(PR_intervals)) else np.nanstd(PR_intervals) / np.nanmean(PR_intervals),
                            "0_Coeff_QT_intervals": 0 if np.isnan(np.nanmean(QT_intervals)) else np.nanstd(QT_intervals) / np.nanmean(QT_intervals),
                            "0_std_pp_intervals": 0 if np.isnan(np.nanstd(pp_intervals)) else np.nanstd(pp_intervals),
                            "0_std_QRS_Interval": 0 if np.isnan(np.nanstd(qrs_intervals)) else np.nanstd(qrs_intervals),
                            "0_HRV_SDNN": 0 if np.isnan(hrv_metrics['HRV_SDNN'].values[0]) else hrv_metrics['HRV_SDNN'].values[0],
                            "0_HRV_RMSSD": 0 if np.isnan(hrv_metrics['HRV_RMSSD'].values[0]) else hrv_metrics['HRV_RMSSD'].values[0],
                            "0_HRV_pNN20": 0 if np.isnan(hrv_metrics['HRV_pNN20'].values[0]) else hrv_metrics['HRV_pNN20'].values[0],
                        }
                        feture = featurenet|features
                        features_df = pd.concat([features_df, pd.DataFrame(feture, index=[0])], ignore_index=False)
                
                except Exception as e:
                    print(f"Error processing patient {patient_folder}: {e}")

features_df.to_csv(output_path, index=False)

