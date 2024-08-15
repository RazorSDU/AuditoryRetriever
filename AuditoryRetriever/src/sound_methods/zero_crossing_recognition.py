# zero_crossing_recognition.py

import numpy as np
import librosa

class ZeroCrossingRecognition:
    @staticmethod
    def extract_zero_crossing_features(sounds):
        features = []
        for audio_data, sample_rate in sounds:
            if len(audio_data) == 0:
                print("Warning: Empty audio data encountered.")
                continue
            zero_crossing = librosa.feature.zero_crossing_rate(y=audio_data)
            zero_crossing_mean = np.mean(zero_crossing)
            features.append(zero_crossing_mean)
        return features
