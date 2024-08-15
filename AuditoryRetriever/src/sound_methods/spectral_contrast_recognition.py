# spectral_contrast_recognition.py

import numpy as np
import librosa

class SpectralContrastRecognition:
    @staticmethod
    def extract_spectral_contrast_features(sounds):
        features = []
        for audio_data, sample_rate in sounds:
            if len(audio_data) == 0:
                print("Warning: Empty audio data encountered.")
                continue
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.append(spectral_contrast_mean)
        return features
