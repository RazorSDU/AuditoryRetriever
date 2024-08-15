# sound_recognition.py

import numpy as np
import os
import librosa
import noisereduce as nr
import pyaudio
from sklearn.svm import SVC
import time
from collections import deque
from sound_methods.fft_recognition import FFTRecognition
from sound_methods.mfcc_recognition import MFCCRecognition
from sound_methods.spectral_contrast_recognition import SpectralContrastRecognition
from sound_methods.chroma_recognition import ChromaRecognition
from sound_methods.zero_crossing_recognition import ZeroCrossingRecognition
from sound_methods.cnn_recognition import CNNRecognition

class SoundRecognizer:
    def __init__(self, sound_directories, noise_reduction_strength=0.5, 
                 use_fft=False, use_cnn=False, use_mfcc=False, use_spectral_contrast=False, 
                 use_chroma_features=False, use_zero_crossing=False, n_fft=4096, hop_length=1024):
        self.noise_reduction_strength = noise_reduction_strength
        self.use_fft = use_fft
        self.use_cnn = use_cnn
        self.use_mfcc = use_mfcc
        self.use_spectral_contrast = use_spectral_contrast
        self.use_chroma_features = use_chroma_features
        self.use_zero_crossing = use_zero_crossing
        self.max_length = 0
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.svm_classifiers = {}
        self.amplitude_threshold = 0
        self.freq_low_threshold = 0 # Unused
        self.freq_high_threshold = 0 # Unused
        self.energy_threshold = 0 # Unused
        self.chunk_duration = 0.01 # This defines the accuracy of trim_silence
        self.trim_silence_counter = 0
        self.last_recognition_time = 0
        self.result_buffer = deque(maxlen=3)
        
        self.frame_size = int(round(44100 * self.chunk_duration))
        # Exception handling to ensure the calculated frame_size is a whole number
        if self.frame_size != 44100 * self.chunk_duration:
            raise ValueError(f"Calculated frame_size is not a whole number: {44100 * self.chunk_duration}. Please check the chunk_duration value.")

        # Initialize CNN recognizer
        self.fft_recognizer = FFTRecognition()
        # self.cnn_recognizer = CNNRecognition(sound_directories)
        
        # Load sounds and calculate features
        self.sounds = self.load_sounds(sound_directories)
        self.sound_features = self.extract_all_features()
        self.avg_features = self.calculate_average_features()
        self.train_svm_classifier()

        
    #-- -------- --#
    #-- TRAINING --#
    #-- -------- --#

    def load_sounds(self, sound_directories):
        """Load sound files from directories."""
        sounds = {}
        for sound_label, directories in sound_directories.items():
            sounds[sound_label] = []
            for directory in directories:
                sounds[sound_label].extend(self.load_wav_files(directory, sound_label))
        return sounds

    def load_wav_files(self, directory, sound_label):
        """Load WAV files from a directory, apply noise reduction, and trim silence."""
        sounds = []
        
        if sound_label.lower() == "silence":
            self.amplitude_threshold = self.calculate_amplitude_threshold(directory, self.chunk_duration)
            # self.freq_low_threshold, self.freq_high_threshold = self.calculate_frequency_threshold(directory)
            # self.energy_threshold = self.calculate_energy_threshold(directory)
        
        for idx, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, filename)
                y, sr = librosa.load(filepath, sr=None)
                if '/Silence' not in directory:
                    # y = self.reduce_noise(y, sr, self.noise_reduction_strength)
                    y = self.trim_silence(y, self.amplitude_threshold, self.chunk_duration)
                if len(y) > 0:
                    sounds.append((y, sr))
                    print(f"{sound_label}: {filename} loaded")
        return sounds


    def extract_all_features(self):
        """Extract features for all sounds."""
        features = {label: {'fft': [], 'mfcc': [], 'spectral_contrast': [], 'chroma': [], 'zero_crossing': [], 'cnn': []} for label in self.sounds.keys()}
        for label, sounds in self.sounds.items():
            features[label]['fft'] = self.fft_recognizer.extract_fft_features(sounds, n_fft=self.n_fft, hop_length=self.hop_length)
            features[label]['mfcc'] = MFCCRecognition.extract_mfcc_features(sounds, n_mfcc=13)
            features[label]['spectral_contrast'] = SpectralContrastRecognition.extract_spectral_contrast_features(sounds)
            features[label]['chroma'] = ChromaRecognition.extract_chroma_features(sounds)
            features[label]['zero_crossing'] = ZeroCrossingRecognition.extract_zero_crossing_features(sounds)
            
            # Calculate max length for FFT features and pad if necessary
            # self.max_length = max(self.max_length, self.calculate_max_length(features[label]['fft']))
            # features[label]['fft'] = [np.pad(yf, (0, self.max_length - len(yf)), 'constant') for yf in features[label]['fft']]        

        return features


    def calculate_max_length(self, features):
        """Calculate the maximum length of the feature arrays."""
        return max(len(feature) for feature in features)

    def calculate_average_features(self):
        """Calculate the average features for each sound label."""
        avg_features = {}
        for label, features in self.sound_features.items():
            avg_features[label] = {}
            
            # Determine the maximum length for FFT features
            max_length = max(len(fft) for fft in features['fft'])

            # Pad or truncate all FFT features to this length
            padded_fft_features = [np.pad(fft, (0, max_length - len(fft)), 'constant') if len(fft) < max_length else fft[:max_length] for fft in features['fft']]
            
            avg_features[label]['fft'] = np.mean(padded_fft_features, axis=0)
            

            avg_features[label]['mfcc'] = np.mean([np.mean(mfcc, axis=1) for mfcc in features['mfcc']], axis=0)
            avg_features[label]['spectral_contrast'] = np.mean(features['spectral_contrast'], axis=0)
            avg_features[label]['chroma'] = np.mean(features['chroma'], axis=0)
            avg_features[label]['zero_crossing'] = np.mean(features['zero_crossing'], axis=0)
        return avg_features

    def train_svm_classifier(self):
        """Train an SVM classifier using positive and negative samples for each sound label."""
        self.svm_classifiers = {}

        for current_label in self.sounds.keys():
            positive_samples = self.sound_features[current_label]['fft']

            # Gather all other labels as negative samples
            negative_samples = []
            for other_label, features in self.sound_features.items():
                if other_label != current_label:
                    negative_samples.extend(features['fft'])

            # Determine the maximum length for padding/truncation
            max_length = max(max(len(f) for f in positive_samples), max(len(f) for f in negative_samples))

            # Pad or truncate all positive and negative samples to this max_length
            positive_samples = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f in positive_samples]
            negative_samples = [np.pad(f, (0, max_length - len(f)), 'constant') if len(f) < max_length else f[:max_length] for f in negative_samples]

            # Prepare the training data
            training_data = np.array(positive_samples + negative_samples)
            labels = np.array([1] * len(positive_samples) + [0] * len(negative_samples))  # 1 for positive, 0 for negative

            # Train the SVM classifier for this specific label
            svm_classifier = SVC(kernel='linear', probability=True)
            svm_classifier.fit(training_data, labels)

            # Store the trained SVM classifier for the current label
            self.svm_classifiers[current_label] = svm_classifier


    #-- ------- --#
    #-- RUNTIME --#
    #-- ------- --#

    def recognize_from_data(self, data, sample_rate):
        """Recognize the sound label for the given raw audio data."""
        
        # Convert 16-bit integer audio data to floating-point values in the range [-1.0, 1.0]
        audio_data = data.astype(np.float32) / 32768.0
        
        if len(audio_data) == 0:
            return {label: 0 for label in self.avg_features}
        
        # Check if enough time (0.01 seconds) has passed to call recognize_sound
        current_time = time.time()
        if current_time - self.last_recognition_time >= 0.1:
            # Call the recognize_sound function and store the result
            result = self.recognize_sound(audio_data, sample_rate)
            self.result_buffer.append(result)
            self.last_recognition_time = current_time
        
        # If the buffer is empty (initial run), return a default dictionary
        if not self.result_buffer:
            return {label: 0 for label in self.avg_features}
        
        # Compute the average of the last 5 results
        averaged_result = {}
        for label in self.avg_features:
            averaged_result[label] = np.mean([res['fft'][label] for res in self.result_buffer])
        
        return {'fft': averaged_result}

    def recognize_sound(self, audio_data, sample_rate):
        """Recognize the sound label with a detailed output integrating negative sampling."""
        processed_audio_data = self.preprocess_audio(audio_data, sample_rate)
        extracted_features = self.extract_features(processed_audio_data, sample_rate)
        return self.predict_sound_SVM(extracted_features)
        # similarity_scores = self.calculate_similarity_scores(extracted_features)
        # return self.calculate_label_specific_scores_with_neg_sampling(similarity_scores)

    def preprocess_audio(self, audio_data, sample_rate):
        """Preprocess the audio data by reducing noise and trimming silence."""
        audio_data = self.trim_silence(audio_data, self.amplitude_threshold, self.chunk_duration)
        return audio_data

    def extract_features(self, audio_data, sample_rate):
        """Extract various features from the audio data."""
        features = {}
        if self.use_fft:
            fft_feature = self.fft_recognizer.extract_fft_feature_single(audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
            features['fft'] = fft_feature
        if self.use_mfcc:
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features['mfcc'] = np.mean(mfcc, axis=1)
        if self.use_spectral_contrast:
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
            features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
        if self.use_chroma_features:
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features['chroma'] = np.mean(chroma, axis=1)
        if self.use_zero_crossing:
            zero_crossing = librosa.feature.zero_crossing_rate(y=audio_data)
            features['zero_crossing'] = np.mean(zero_crossing)
        return features

    def calculate_similarity_scores(self, extracted_features):
        """Calculate similarity scores between extracted features and average features."""
        similarity_scores = {}
        feature_weights = self.get_feature_weights()

        for label, features in self.avg_features.items():
            if self.use_fft:
                fft_score = self.cosine_similarity(features['fft'], extracted_features['fft'])
                similarity_scores.setdefault('fft', {})[label] = fft_score * 100
                
            if self.use_mfcc:
                mfcc_score = self.cosine_similarity(features['mfcc'], extracted_features['mfcc'])
                mfcc_score_normalized = self.normalize_similarity(mfcc_score)
                similarity_scores.setdefault('mfcc', {})[label] = mfcc_score_normalized * 100
                
            if self.use_spectral_contrast:
                spectral_contrast_score = self.cosine_similarity(features['spectral_contrast'], extracted_features['spectral_contrast'])
                spectral_contrast_score_normalized = self.normalize_similarity(spectral_contrast_score)
                similarity_scores.setdefault('spectral_contrast', {})[label] = spectral_contrast_score_normalized * 100
                
            if self.use_chroma_features:
                chroma_score = self.cosine_similarity(features['chroma'], extracted_features['chroma'])
                chroma_score_normalized = self.normalize_similarity(chroma_score)
                similarity_scores.setdefault('chroma', {})[label] = chroma_score_normalized * 100
                
            if self.use_zero_crossing:
                zero_crossing_score = self.cosine_similarity(np.array([features['zero_crossing']]), np.array([extracted_features['zero_crossing']]))
                zero_crossing_score_normalized = self.normalize_similarity(zero_crossing_score)
                similarity_scores.setdefault('zero_crossing', {})[label] = zero_crossing_score_normalized * 100
                
        return similarity_scores

    def predict_sound_SVM(self, extracted_features):
        """Predict the sound label probabilities for given audio data using trained SVM classifiers."""

        # Prepare the dictionary to store prediction probabilities for each label
        label_specific_scores = {}
        if self.use_fft:
            for label, svm_classifier in self.svm_classifiers.items():
                # Predict the probability for this label using the extracted features
                probability = svm_classifier.predict_proba([extracted_features['fft']])[0][1]  # Get probability for class 1 (positive class)
                label_specific_scores[label] = probability * 100  # Scale to percentage

        # Return the result in the format {'fft': {...}} FOR TESTING
        return {'fft': label_specific_scores}


    def get_feature_weights(self):
        """Get weights for each feature based on their usage."""
        feature_count = sum([self.use_fft, self.use_mfcc, self.use_spectral_contrast, self.use_chroma_features, self.use_zero_crossing, self.use_cnn])
        base_weight = 1.0 / feature_count if feature_count > 0 else 0
        return {
            'fft': base_weight if self.use_fft else 0,
            'mfcc': base_weight if self.use_mfcc else 0,
            'spectral_contrast': base_weight if self.use_spectral_contrast else 0,
            'chroma': base_weight if self.use_chroma_features else 0,
            'zero_crossing': base_weight if self.use_zero_crossing else 0,
            'cnn': base_weight if self.use_cnn else 0
        }


    #-- -------------- --#
    #-- RUNTIME ADDONS --#
    #-- -------------- --#

    def cosine_similarity(self, vector_a, vector_b):
        min_len = min(len(vector_a), len(vector_b))
        if np.linalg.norm(vector_a[:min_len]) == 0 or np.linalg.norm(vector_b[:min_len]) == 0:
            return 0  # Return 0 similarity if any vector is zero to avoid NaN
        normalized_a = vector_a[:min_len] / np.linalg.norm(vector_a[:min_len])
        normalized_b = vector_b[:min_len] / np.linalg.norm(vector_b[:min_len])
        return np.dot(normalized_a, normalized_b)

    def normalize_similarity(self, score):
        """Normalize the similarity score to a range of [0, 1]. CURRENTLY OPSELETE"""
        return score
    
    def play_audio(self, audio_data, channels=1, rate=44100):
        p = pyaudio.PyAudio()
        stream_out = p.open(format=pyaudio.paFloat32,
                            channels=channels,
                            rate=rate,
                            output=True)
        
        stream_out.write(audio_data.astype(np.float32).tobytes())
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()

    def trim_silence(self, audio_data, amplitude_threshold, chunk_duration, min_length=0.1):
        """Trim silence from the beginning and end of the audio data based on amplitude, adding a buffer around detected sounds."""
        
        # Ensure the audio data is in float32 format for consistency
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Calculate the number of frames that make up a chunk of 0.06 seconds
        frames_per_chunk = int(0.01 / chunk_duration)
        
        # Break the audio data into frames
        frames = [audio_data[i:i+self.frame_size] for i in range(0, len(audio_data), self.frame_size)]
        
        # Calculate amplitude for each frame
        amplitude_values = np.array([np.max(np.abs(frame)) for frame in frames])
        
        # Check if all amplitudes are below the threshold
        if np.all(amplitude_values < amplitude_threshold):
            return audio_data[:]  # Return the original audio data without trimming

        # Initialize a list to keep track of frames to keep
        keep_indices = np.zeros(len(amplitude_values), dtype=bool)
        
        # Iterate over chunks of frames
        for i in range(len(amplitude_values)):
            if amplitude_values[i] >= amplitude_threshold:
                # Mark this chunk and add buffer before and after it
                start_idx = max(0, i - 1)  # Include 1 chunks before
                end_idx = min(len(amplitude_values), i + 2)  # Include 1 chunks after
                keep_indices[start_idx:end_idx] = True
        
        # If no non-silent frames are found, return an empty array
        if not np.any(keep_indices):
            return np.array([], dtype=np.float32)
        
        # Determine the audio to keep based on the marked frames
        start = np.argmax(keep_indices) * self.frame_size
        end = (len(keep_indices) - np.argmax(keep_indices[::-1]) - 1) * self.frame_size + self.frame_size
        end = min(end, len(audio_data))
        
        # Ensure the trimmed audio is at least min_length seconds long
        if (end - start) < min_length * self.frame_size:
            return audio_data[:]
        
        trimmed_audio = audio_data[start:end]

        # Increment the counter
        self.trim_silence_counter += 1
        
        # After the first 500 calls, print the amplitudes for the next 3 calls
        if 73 <= self.trim_silence_counter <= 76:
            
            # Calculate total chunks and total time for both original and trimmed audio
            total_chunks_original = len(audio_data) // self.frame_size
            total_chunks_trimmed = len(trimmed_audio) // self.frame_size
            
            total_time_original = len(audio_data) / 44100.0
            total_time_trimmed = len(trimmed_audio) / 44100.0
            
            print("Analyzing original audio...")
            for i in range(total_chunks_original):
                segment = audio_data[i * self.frame_size:(i + 1) * self.frame_size]
                amplitude = np.max(np.abs(segment))
                current_time = (i + 1) * self.chunk_duration
                print(f"Time: {current_time:.2f}/{total_time_original:.2f} sec | Amplitude: {amplitude * 32767:.2f}")
            
            print("Analyzing trimmed audio...")
            for i in range(total_chunks_trimmed):
                segment = trimmed_audio[i * self.frame_size:(i + 1) * self.frame_size]
                amplitude = np.max(np.abs(segment))
                current_time = (i + 1) * self.chunk_duration
                print(f"Time: {current_time:.2f}/{total_time_trimmed:.2f} sec | Amplitude: {amplitude * 32767:.2f}")

        # Every 50th time, print the length before and after trimming
        if self.trim_silence_counter % 50 == 0:
            original_length = len(audio_data)
            trimmed_length = len(trimmed_audio)
            
            # Convert lengths to seconds and milliseconds
            original_seconds = original_length / 44100
            trimmed_seconds = trimmed_length / 44100
            
            original_time = f"{int(original_seconds)}:{int((original_seconds % 1) * 1000):03d}"
            trimmed_time = f"{int(trimmed_seconds)}:{int((trimmed_seconds % 1) * 1000):03d}"
            
            avg_amplitude = np.mean(amplitude_values)
            max_amplitude = np.max(amplitude_values)
            
            print(f"Trim: {original_time} -> {trimmed_time}")
            print(f"Avg amp: {avg_amplitude:.5f} | Max amp: {max_amplitude:.5f}")
        
        #self.play_audio(trimmed_audio)
        
        return trimmed_audio

    # Function to calculate amplitude threshold
    def calculate_amplitude_threshold(self, silence_directory, chunk_duration):
        """Calculate the threshold for silence based on the silence files."""
        silence_amplitude_values = []
        
        for filename in os.listdir(silence_directory):
            if filename.endswith(".wav"):
                filepath = os.path.join(silence_directory, filename)
                y, sr = librosa.load(filepath, sr=None)
                
                # Ensure the audio data is in float32 format
                if y.dtype != np.float32:
                    y = y.astype(np.float32)
                
                # Calculate chunk size based on the sampling rate and chunk duration
                chunk_size = int(sr * chunk_duration)
                
                # Process the audio in chunks and calculate the amplitude
                for i in range(0, len(y), chunk_size):
                    segment = y[i:i+chunk_size]
                    if len(segment) > 0:
                        amplitude = np.max(np.abs(segment))
                        silence_amplitude_values.append(amplitude)
        
        # Calculate the average and maximum amplitude values
        avg_amplitude = np.mean(silence_amplitude_values)
        max_amplitude = np.max(silence_amplitude_values)
        print(f"max Amplitude Threshold: {max_amplitude:.5f}")
        print(f"avg Amplitude Threshold: {avg_amplitude:.5f}") 

        # Check if the maximum amplitude is an extreme outlier compared to the average
        amp_multiplier = 2  # ::PARAMETER:: Adjust the multiplier as needed
        if max_amplitude > avg_amplitude * amp_multiplier:
            # Remove the extreme outlier and recalculate the max amplitude
            silence_amplitude_values = [amp for amp in silence_amplitude_values if amp < avg_amplitude * amp_multiplier]
            max_amplitude = np.max(silence_amplitude_values)
        
        print(f"max after removing anomalies: {max_amplitude:.5f}")

        # Set the silence threshold slightly above the maximum amplitude value of the (filtered) silence files
        return max_amplitude * 2.2



    #-- ------------- --#
    #-- UNUSED ADDONS --#
    #-- ------------- --#

    def calculate_frequency_threshold(self, silence_directory):
        """Calculate the frequency thresholds based on the silence files."""
        silence_freq_values = []

        for filename in os.listdir(silence_directory):
            if filename.endswith(".wav"):
                filepath = os.path.join(silence_directory, filename)
                y, sr = librosa.load(filepath, sr=None)
                
                # Calculate the spectral centroid (represents the "center of mass" of the frequencies)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                silence_freq_values.extend(spectral_centroid[0])

        # Calculate the average and maximum frequency values
        avg_freq = np.mean(silence_freq_values)
        max_freq = np.max(silence_freq_values)
        min_freq = np.min(silence_freq_values)

        # Apply a margin to set frequency thresholds
        margin = 1.1  # ::PARAMETER:: Adjust the margin as needed
        high_freq_threshold = max_freq * margin
        low_freq_threshold = min_freq * margin

        return low_freq_threshold, high_freq_threshold

    def reduce_noise(self, audio_data, sample_rate, strength=0.5, noise_clip=None):
            """Reduce noise in the audio data using the noisereduce library."""
            if noise_clip is None:
                noise_clip = audio_data
            return nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=strength, y_noise=noise_clip)
    
    def calculate_energy_threshold(self, silence_directory):
        """Calculate the energy threshold for silence based on the silence files."""
        silence_energy_values = []
        
        for filename in os.listdir(silence_directory):
            if filename.endswith(".wav"):
                filepath = os.path.join(silence_directory, filename)
                y, sr = librosa.load(filepath, sr=None)
                
                # Calculate the energy of the audio signal (sum of squares)
                energy_value = np.sum(y**2)
                silence_energy_values.append(energy_value)
        
        # Calculate the average and maximum energy values
        avg_energy = np.mean(silence_energy_values)
        max_energy = np.max(silence_energy_values)

        # Check if the maximum energy is an extreme outlier compared to the average
        energy_multiplier = 1.5  # ::PARAMETER:: Adjust the multiplier as needed
        if max_energy > avg_energy * energy_multiplier:
            # Remove the extreme outlier and recalculate the max energy
            silence_energy_values = [energy for energy in silence_energy_values if energy < avg_energy * energy_multiplier]
            max_energy = np.max(silence_energy_values)

        # Set the silence threshold slightly above the maximum energy value of the (filtered) silence files
        return max_energy * 1.05  # ::PARAMETER:: Add a small margin to the calculated threshold