# fft_recognition.py

import numpy as np
import librosa

class FFTRecognition:
    def extract_fft_features(self, sounds, n_fft=4096, hop_length=1024):
        features = []
        n_fft_init = n_fft
        for audio_data, sample_rate in sounds:
            if len(audio_data) == 0:  # Check for empty audio data
                print("Warning: Empty audio data encountered.")
                continue
            
            # Adjust n_fft if it's larger than the audio data length
            if len(audio_data) < n_fft:
                n_fft = len(audio_data)

            # Perform STFT to extract time-frequency features
            stft = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, window='hann', center=False))
            
            # Calculate the mean across the time axis (averaging over all frames)
            fft_mean = np.mean(stft, axis=1)
            
            # Normalize the FFT features if necessary
            fft_mean = self.normalize_fft_features(fft_mean)
            
            features.append(fft_mean)
            
            n_fft = n_fft_init
        return features

    def extract_fft_feature_single(self, audio_data, n_fft=4096, hop_length=1024):
        """Extract FFT features from a single audio_data array."""
        if len(audio_data) == 0:  # Check for empty audio data
            print("Warning: Empty audio data encountered.")
            return []

        # Adjust n_fft if it's larger than the audio data length
        if len(audio_data) < n_fft:
            n_fft = len(audio_data)

        # Perform STFT to extract time-frequency features
        stft = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, window='hann', center=False))
        
        # Calculate the mean across the time axis (averaging over all frames)
        fft_mean = np.mean(stft, axis=1)
        
        # Normalize the FFT features if necessary
        fft_mean = self.normalize_fft_features(fft_mean)
        
        return fft_mean

    def normalize_fft_features(self, fft_features):
        """Normalize FFT features to prevent scale issues."""
        norm_fft = np.linalg.norm(fft_features)
        if norm_fft == 0:
            return fft_features
        return fft_features / norm_fft




