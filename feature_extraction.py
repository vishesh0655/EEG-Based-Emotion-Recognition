

import os
import numpy as np
import pandas as pd
import scipy.signal

FS = 128   

CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2',  'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

BANDS = {
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 45),
}

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IN_PATH   = os.path.join(BASE_DIR, 'DREAMER_preprocessed.npy')
OUT_PATH  = os.path.join(BASE_DIR, 'DREAMER_features.csv')


def band_power(channel_signal, low, high, fs=FS):
   
    freqs, psd = scipy.signal.welch(channel_signal, fs=fs, nperseg=len(channel_signal))
    band_mask  = (freqs >= low) & (freqs <= high)
    return float(np.mean(psd[band_mask]))


def extract_features(window):
    
    features = {}
    for band_name, (low, high) in BANDS.items():
        for ch_idx, ch_name in enumerate(CHANNELS):
            ch_signal = window[:, ch_idx]
            features[f'{band_name}_{ch_name}'] = band_power(ch_signal, low, high)
    return features


def main():
    print(f"Loading preprocessed data from: {IN_PATH}")
    data = np.load(IN_PATH, allow_pickle=True).item()

    windows       = data['windows']        
    subjects      = data['subject']
    trials        = data['trial']
    window_nums   = data['window_num']
    valence_raw   = data['valence']
    arousal_raw   = data['arousal']
    valence_label = data['valence_label']
    arousal_label = data['arousal_label']

    n_windows = len(windows)
    print(f"Total windows to process: {n_windows}")

    all_rows = []

    for i in range(n_windows):
        if (i + 1) % 5000 == 0:
            print(f"  Processing window {i+1}/{n_windows} ...")

        window = windows[i]   

        feats = extract_features(window)

        row = {
            'subject': subjects[i],
            'trial':   trials[i],
            'window':  window_nums[i],
            **feats,
            'valence':       valence_raw[i],
            'arousal':       arousal_raw[i],
            'valence_label': valence_label[i],
            'arousal_label': arousal_label[i],
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nDone! Saved {len(df)} rows : {OUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Features: {[c for c in df.columns if '_' in c and c not in ['valence_label','arousal_label']][:5]} ... (56 total)")


if __name__ == '__main__':
    main()