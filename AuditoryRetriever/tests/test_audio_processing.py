import librosa
import numpy as np
import sounddevice as sd
import keyboard
import time
import psutil
import os
import threading
from queue import Queue

# Set initial sample rate and other quality factors
sample_rate = int(input("Enter initial sample rate (default 22050): ") or "22050")
channels = 1  # Mono recording by default

# Initialize global variables
recorded_audio = None
recording = False
record_start_time = None
record_end_time = None
audio_duration = None
listening = False
audio_queue = Queue()

# Function to display audio information
def display_audio_info(y, sr):
    duration = len(y) / sr
    print(f"Audio Duration: {duration:.2f} seconds")
    print(f"Sample Rate: {sr}")
    print(f"Number of Samples: {len(y)}")

# Function to play audio with countdown timer
def play_audio(y, sr):
    global audio_duration

    def countdown():
        start_time = time.time()
        while time.time() - start_time < audio_duration:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            milliseconds = int((elapsed_time - int(elapsed_time)) * 100)
            print(f"\r{minutes:02}:{seconds:02}:{milliseconds:02} / {audio_duration:.2f}", end="")
            time.sleep(0.05)
        print("\rAudio playback finished. You can play the sound again.")
    
    playback_thread = threading.Thread(target=countdown)
    playback_thread.start()
    sd.play(y[:int(audio_duration * sr)], sr)
    sd.wait(audio_duration)

def start_recording():
    global recorded_audio, recording, record_start_time
    if not recording:
        print("Recording started... Begin talking now.")
        recording = True
        record_start_time = time.time()
        recorded_audio = sd.rec(int(10 * sample_rate), samplerate=sample_rate, channels=channels)
    else:
        print("Already recording...")

def stop_recording():
    global recording, record_end_time, audio_duration
    if recording:
        sd.stop()
        recording = False
        record_end_time = time.time()
        audio_duration = record_end_time - record_start_time
        print(f"Recording stopped. Duration: {audio_duration:.2f} seconds")
        display_audio_info(recorded_audio[:int(audio_duration * sample_rate)], sample_rate)
        analyze_audio(recorded_audio[:int(audio_duration * sample_rate)], sample_rate)
    else:
        print("Not currently recording...")

# Function to compute and display performance indicators
def analyze_audio(y, sr):
    # Ensure the audio buffer contains finite values
    y = np.nan_to_num(y)

    start_time = time.time()
    S = librosa.feature.melspectrogram(y=y.flatten(), sr=sr)
    end_time = time.time()
    compute_time = end_time - start_time

    start_memory = psutil.Process(os.getpid()).memory_info().rss
    mfcc = librosa.feature.mfcc(y=y.flatten(), sr=sr, n_mfcc=13)
    end_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_usage = end_memory - start_memory

    print("\nPerformance Indicators:")
    print(f"Time to compute: {compute_time:.4f} seconds")
    print(f"Memory used: {memory_usage / (1024 * 1024):.2f} MB")

def change_sample_rate():
    global sample_rate
    sample_rate = int(input("Enter new sample rate: "))
    print(f"Sample rate changed to {sample_rate}")

def change_channels():
    global channels
    channels = int(input("Enter number of channels (1 for mono, 2 for stereo): "))
    print(f"Channels changed to {channels}")

def monitor_memory_and_latency():
    global listening
    process = psutil.Process(os.getpid())
    start_time = time.time()
    while listening:
        current_memory = process.memory_info().rss / (1024 * 1024)  # Memory usage in MB
        elapsed_time = time.time() - start_time
        print(f"\rMemory Usage: {current_memory:.2f} MB, Elapsed Time: {elapsed_time:.2f} seconds", end="")
        time.sleep(0.5)

def start_listening():
    global listening
    if not listening:
        print("Listening started...")
        listening = True
        stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback)
        stream.start()
        threading.Thread(target=monitor_memory_and_latency).start()
        threading.Thread(target=play_from_queue).start()

def stop_listening():
    global listening
    if listening:
        print("\nListening stopped.")
        listening = False

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    audio_queue.put(indata.copy())

def play_from_queue():
    global listening
    while listening:
        if not audio_queue.empty():
            data = audio_queue.get()
            sd.play(data, sample_rate)
            sd.wait()

# Listen for key presses to record, stop recording, change settings, and play audio
print("Press 'a' to start recording, 'b' to stop recording, and 'z' to play the recorded audio.")
print("Press 'r' to change sample rate, 'c' to change channels.")
print("Press 'q' to start real-time listening, 'w' to stop real-time listening.")

keyboard.add_hotkey('a', start_recording)
keyboard.add_hotkey('b', stop_recording)
keyboard.add_hotkey('z', lambda: play_audio(recorded_audio, sample_rate) if recorded_audio is not None else print("No audio recorded"))
keyboard.add_hotkey('r', change_sample_rate)
keyboard.add_hotkey('c', change_channels)
keyboard.add_hotkey('q', start_listening)
keyboard.add_hotkey('w', stop_listening)

# Keep the script running to listen for key presses
keyboard.wait('esc')



