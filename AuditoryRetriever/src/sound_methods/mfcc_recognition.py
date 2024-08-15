# mfcc_recognition.py

import numpy as np
import librosa

class MFCCRecognition:
    @staticmethod
    def extract_mfcc_features(sounds, n_mfcc=13, sr=None, n_fft=512, hop_length=512, win_length=None, window='hann', center=True, pad_mode='reflect'):
        mfcc_features = []
        for audio, original_sr in sounds:
            if len(audio) == 0:
                print("Warning: Empty audio data encountered.")
                continue
            if sr is not None:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=sr)
            else:
                sr = original_sr
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
            mfcc_features.append(mfcc)
        return mfcc_features
