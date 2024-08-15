# chroma_recognition.py

import numpy as np
import librosa

class ChromaRecognition:
    @staticmethod
    def extract_chroma_features(sounds):
        features = []
        for audio_data, sample_rate in sounds:
            if len(audio_data) == 0:
                print("Warning: Empty audio data encountered.")
                continue
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            features.append(chroma_mean)
        return features
