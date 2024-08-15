# cnn_recognition.py

import noisereduce as nr
import os
import numpy as np
import librosa
from tensorflow.keras import layers, models

class CNNRecognition:
    def __init__(self, sound_directories):
        self.label_map = {label: idx for idx, label in enumerate(sound_directories.keys())}
        self.model = self.train_cnn_model(sound_directories)

    def train_cnn_model(self, sound_directories):
        spectrograms, labels = self.prepare_spectrogram_data(sound_directories)
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(sound_directories), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(spectrograms, labels, epochs=10)
        return model

    def prepare_spectrogram_data(self, sound_directories):
        spectrograms = []
        labels = []
        for label, directories in sound_directories.items():
            for directory in directories:
                for filename in os.listdir(directory):
                    if filename.endswith(".wav"):
                        filepath = os.path.join(directory, filename)
                        y, sr = librosa.load(filepath, sr=None)
                        y = self.reduce_noise(y, sr)
                        y = self.trim_silence(y)
                        if len(y) > 0:
                            spectrogram = self.create_spectrogram(y, sr)
                            spectrograms.append(spectrogram)
                            labels.append(self.label_map[label])
        return np.array(spectrograms), np.array(labels)

    @staticmethod
    def create_spectrogram(y, sr):
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=512)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = np.expand_dims(spectrogram_db, axis=-1)
        spectrogram_db = np.resize(spectrogram_db, (128, 128, 1))
        return spectrogram_db

    @staticmethod
    def reduce_noise(audio_data, sample_rate):
        return nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.5)

    @staticmethod
    def trim_silence(audio_data, threshold=0.01, frame_size=512):
        rms = lambda x: np.sqrt(np.mean(x**2))
        frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
        rms_values = np.array([rms(frame) for frame in frames])
        non_silent_indices = np.where(rms_values > threshold)[0]
        if len(non_silent_indices) == 0:
            return np.array([])
        start = non_silent_indices[0] * frame_size
        end = non_silent_indices[-1] * frame_size + frame_size
        return audio_data[start:end]

    def predict_with_cnn(self, audio_data, sample_rate):
        spectrogram = self.create_spectrogram(audio_data, sample_rate)
        spectrogram = np.expand_dims(spectrogram, axis=0)
        prediction = self.model.predict(spectrogram, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        label_map = {idx: label for label, idx in self.label_map.items()}
        predicted_label_name = label_map[predicted_label]
        return predicted_label_name, prediction[0][predicted_label]
