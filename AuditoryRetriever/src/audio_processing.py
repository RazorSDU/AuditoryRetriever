# audio_processing.py

import pyaudio
import time
import numpy as np
import os
from datetime import datetime
import soundfile as sf
import panel as pn
from src.debugging import Debugger
from src.sound_recognition import SoundRecognizer

pn.extension()

class AudioProcessor:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 2048

    def __init__(self, debugger=None):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        self.stream_out = self.p.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      output=True)
        self.audio_enabled = False
        self.recording = False
        self.record_buffer = []
        self.record_start_time = None
        self.previous_estimated_size = 0
        self.previous_time = None
        self.size_velocity_list = []
        self.size_velocity_window = 10
        self.debugger = debugger or Debugger()
        self.toggle_soundrecognition = False
        
        # Sound Recognizer Initialization
        sound_directories = {
            'silence': ["data/recordings/Mic1/Silence"], # MUST BE FIRST!
            'mouse_click': ["data/recordings/Mic1/Mouse_Click"],
            'clap': ["data/recordings/Mic1/Clap"],
            'desk_tap': ["data/recordings/Mic1/Desk_tap"],
            'keyboard_space': ["data/recordings/Mic1/Keyboard_Space"],
            'snap': ["data/recordings/Mic1/Snap"]
            # Add more sound directories as needed
        }
        self.sound_recognizer = SoundRecognizer(
            sound_directories=sound_directories, 
            use_fft=False,  
            use_mfcc=False, 
            use_spectral_contrast=False, 
            use_chroma_features=False, 
            use_zero_crossing=False, 
            use_cnn=False, 
            n_fft=4096,  # Use the desired n_fft value
            hop_length=1024  # Use the desired hop_length value
        )
        
        # Panel placeholders
        self.audio_status_placeholder = pn.pane.Markdown("")
        self.recording_status_placeholder = pn.pane.Markdown("")
        self.recording_info_placeholder = pn.pane.Markdown("")
        self.recognition_result_placeholder = {}
    
    def set_use_fft(self, value):
        self.sound_recognizer.use_fft = value
        self.toggle_soundrecognition = value

    def set_use_mfcc(self, value):
        self.sound_recognizer.use_mfcc = value

    def set_use_cnn(self, value):
        self.sound_recognizer.use_cnn = value

    def set_use_spectral_contrast(self, value):
        self.sound_recognizer.use_spectral_contrast = value

    def set_use_chroma_features(self, value):
        self.sound_recognizer.use_chroma_features = value

    def set_use_zero_crossing(self, value):
        self.sound_recognizer.use_zero_crossing = value

    def toggle_audio(self, state=None):
        if state is not None:
            self.audio_enabled = state
        else:
            self.audio_enabled = not self.audio_enabled
        self.audio_status_placeholder.object = f"Audio {'enabled' if self.audio_enabled else 'disabled'}"

    def save_audio(self, data, filename, sr=RATE):
        os.makedirs('data/recordings', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join('data/recordings', f"{timestamp}_{filename}")

        data_np = np.frombuffer(data, dtype=np.int16)

        # Calculate the number of samples for 0.2 seconds
        trim_samples = int(0.2 * sr)

        # Trim the first and last 0.2 seconds
        trimmed_data = data_np[trim_samples:-trim_samples]

        # Check if the remaining audio is at least 0.8 seconds long
        if len(trimmed_data) >= int(0.8 * sr):
            # Save the trimmed audio only if it meets the length requirement
            sf.write(file_path, trimmed_data, sr)
    
    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.record_buffer = []
            self.record_start_time = time.time()
            self.previous_estimated_size = 0
            self.previous_time = time.time()
            self.size_velocity_list = []
            self.recording_status_placeholder.object = "Recording started"

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.record_buffer:
                self.save_audio(b''.join(self.record_buffer), 'recorded_output.wav')
            self.record_buffer = []
            self.previous_estimated_size = 0
            self.record_start_time = None
            self.previous_time = None
            self.size_velocity_list = []
            self.recording_status_placeholder.object = "Recording stopped and saved to 'data/recordings/'"

    def estimate_file_size(self):
        return len(b''.join(self.record_buffer)) / (1024 ** 2)

    def display_recording_info(self):
        # Check if recording is active
        if not self.recording or self.record_start_time is None:
            self.recording_info_placeholder.object = "No recording is currently active."
            if self.recording and self.record_start_time is None:
                self.record_start_time = time.time()
            return

        # Calculate elapsed time
        current_time = time.time()
        elapsed_time = current_time - self.record_start_time
        minutes, seconds = divmod(elapsed_time, 60)
        milliseconds = (elapsed_time % 1) * 1000

        # Estimate current file size (in MB)
        estimated_size_mb = self.estimate_file_size()

        # Calculate size velocity
        if self.previous_time is not None and self.previous_estimated_size is not None:
            time_diff = current_time - self.previous_time
            size_diff = estimated_size_mb - self.previous_estimated_size

            if time_diff > 0:
                size_velocity = size_diff / time_diff
                self.size_velocity_list.append(size_velocity)

                # Maintain a list of velocity values to smooth out fluctuations
                if len(self.size_velocity_list) > self.size_velocity_window:
                    self.size_velocity_list.pop(0)

                # Calculate the smoothed size velocity
                smoothed_velocity = np.mean(self.size_velocity_list)
            else:
                smoothed_velocity = 0.0
        else:
            smoothed_velocity = 0.0

        # Update previous time and size
        self.previous_time = current_time
        self.previous_estimated_size = estimated_size_mb

        # Display the information
        self.recording_info_placeholder.object = f"""
            **Recording Information:**
            - Elapsed Time: {int(minutes)} minutes, {int(seconds)} seconds, {int(milliseconds)} milliseconds
            - Estimated File Size: {estimated_size_mb:.2f} MB
            - File Size Growth Rate: {smoothed_velocity:.2f} MB/s
        """

    def process_audio(self):
        BUFFER_SIZE = int(44100/2)  # 1 second buffer at 44.1 kHz sampling rate because the rate is 44100 that means that 44100 buffer size would be 1 sec
        audio_buffer = np.zeros(BUFFER_SIZE)
        
        def update_audio_buffer(chunk):
            """Updates the audio buffer with new chunk of data."""
            nonlocal audio_buffer
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

        try:
            while True:
                data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                data_np = np.frombuffer(data, dtype=np.int16)
                
                # Update the audio buffer with the latest chunk
                update_audio_buffer(data_np)
    
                if self.audio_enabled:
                    self.stream_out.write(data)
                if self.recording:
                    self.record_buffer.append(data)
                    self.display_recording_info()
                if self.toggle_soundrecognition:
                    # Real-time sound recognition on the buffered data
                    recognition_results = self.sound_recognizer.recognize_from_data(audio_buffer, self.RATE)
                    self.recognition_result_placeholder = recognition_results

        except Exception as e:
            self.recording_info_placeholder.object = f"An error occurred: {e}"
