

import os
import numpy as np
import scipy.io
import scipy.signal


FS = 128       
WINDOW_SIZE = FS        
LABEL_THRESHOLD = 3     

CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2',  'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAT_PATH = os.path.join(BASE_DIR, 'DREAMER.mat')
OUT_PATH = os.path.join(BASE_DIR, 'DREAMER_preprocessed.npy')



def bandpass_filter(signal, low=1.0, high=45.0, fs=FS, order=5):
    
    nyq = fs / 2.0
    b, a = scipy.signal.butter(order, [low / nyq, high / nyq], btype='band')
    return scipy.signal.filtfilt(b, a, signal)



def notch_filter(signal, notch_freq=50.0, fs=FS, quality=30):
    
    b, a = scipy.signal.iirnotch(notch_freq / (fs / 2.0), quality)
    return scipy.signal.filtfilt(b, a, signal)



def baseline_correct(stimuli, baseline):
    
    baseline_mean = np.mean(baseline, axis=0)   
    return stimuli - baseline_mean



def filter_eeg(eeg_data):
    
    filtered = np.zeros_like(eeg_data)
    for ch in range(eeg_data.shape[1]):
        ch_signal = eeg_data[:, ch]
        ch_signal = bandpass_filter(ch_signal)
        ch_signal = notch_filter(ch_signal)
        filtered[:, ch] = ch_signal
    return filtered


def sliding_window(eeg_data, window_size=WINDOW_SIZE):
    
    n_samples = eeg_data.shape[0]
    n_windows = n_samples // window_size
    windows = []
    for w in range(n_windows):
        start = w * window_size
        end   = start + window_size
        windows.append(eeg_data[start:end, :])
    return windows


def main():
    print("Loading DREAMER.mat ...")
    mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
    dreamer  = mat['DREAMER']
    subjects = dreamer.Data      

    all_windows       = []   
    all_subjects      = []
    all_trials        = []
    all_window_nums   = []
    all_valence_raw   = []
    all_arousal_raw   = []
    all_valence_label = []
    all_arousal_label = []

    for sub_idx, subject in enumerate(subjects):
        sub_id = sub_idx + 1
        print(f"  Preprocessing Subject {sub_id}/23 ...")

        eeg            = subject.EEG
        valence_scores = subject.ScoreValence  
        arousal_scores = subject.ScoreArousal   

        for trial_idx in range(18):
            trial_id = trial_idx + 1

            stimuli  = eeg.stimuli[trial_idx]   
            baseline = eeg.baseline[trial_idx]   

            corrected = baseline_correct(stimuli, baseline)

            filtered = filter_eeg(corrected)

            windows = sliding_window(filtered)

            v_raw   = int(valence_scores[trial_idx])
            a_raw   = int(arousal_scores[trial_idx])
            v_label = 'High' if v_raw > LABEL_THRESHOLD else 'Low'
            a_label = 'High' if a_raw > LABEL_THRESHOLD else 'Low'

            for w_num, win in enumerate(windows):
                all_windows.append(win)
                all_subjects.append(sub_id)
                all_trials.append(trial_id)
                all_window_nums.append(w_num + 1)
                all_valence_raw.append(v_raw)
                all_arousal_raw.append(a_raw)
                all_valence_label.append(v_label)
                all_arousal_label.append(a_label)

    
    data_to_save = {
        'windows':       np.array(all_windows),        
        'subject':       np.array(all_subjects),
        'trial':         np.array(all_trials),
        'window_num':    np.array(all_window_nums),
        'valence':       np.array(all_valence_raw),
        'arousal':       np.array(all_arousal_raw),
        'valence_label': np.array(all_valence_label),
        'arousal_label': np.array(all_arousal_label),
        'channels':      CHANNELS,
    }

    np.save(OUT_PATH, data_to_save, allow_pickle=True)

    print(f"\nDone! Saved {len(all_windows)} windows to: {OUT_PATH}")
    print(f"Windows shape: {np.array(all_windows).shape}")


if __name__ == '__main__':
    main()