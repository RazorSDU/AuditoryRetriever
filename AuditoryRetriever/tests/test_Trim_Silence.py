import numpy as np
import pyaudio
import wave
from scipy.fft import fft
import time
import librosa
import os

# Function to trim silence based on amplitude and add buffer chunks
def trim_silence(audio_data, amp_threshold, frame_size, min_length=0.1, chunk_duration=0.01):
    """Trim silence from the beginning and end of the audio data based on amplitude, adding a buffer around detected sounds."""
    
    # Calculate the number of frames that make up a chunk of 0.06 seconds
    frames_per_chunk = int(0.06 / chunk_duration)
    
    # Break the audio data into frames
    frames = [audio_data[i:i+frame_size] for i in range(0, len(audio_data), frame_size)]
    
    # Calculate amplitude for each frame
    amplitude_values = np.array([np.max(np.abs(frame)) for frame in frames])
    
    # Initialize a list to keep track of frames to keep
    keep_indices = np.zeros(len(amplitude_values), dtype=bool)
    
    # Iterate over chunks of frames
    for i in range(len(amplitude_values)):
        if amplitude_values[i] >= amp_threshold:
            # Mark this chunk and add buffer before and after it
            start_idx = max(0, i - 2)  # Include 2 chunks before
            end_idx = min(len(amplitude_values), i + 3)  # Include 2 chunks after
            keep_indices[start_idx:end_idx] = True
    
    # If no non-silent frames are found, return an empty array
    if not np.any(keep_indices):
        return np.array([])
    
    # Determine the audio to keep based on the marked frames
    start = np.argmax(keep_indices) * frame_size
    end = (len(keep_indices) - np.argmax(keep_indices[::-1]) - 1) * frame_size + frame_size
    end = min(end, len(audio_data))
    
    # Ensure the trimmed audio is at least min_length seconds long
    if (end - start) < min_length * frame_size:
        return audio_data[:]
    
    return audio_data[start:end]

# Function to calculate amplitude, frequency, and energy
def analyze_audio_segment(segment, rate):
    amplitude = np.max(np.abs(segment))
    energy = np.sum(segment**2)
    
    # Compute frequency using FFT
    fft_data = np.abs(fft(segment))
    freqs = np.fft.fftfreq(len(segment), 1 / rate)
    peak_freq = freqs[np.argmax(fft_data[:len(fft_data)//2])]
    
    return amplitude, peak_freq, energy

# Function to play audio and analyze it
def play_and_analyze_audio(audio_data, rate, channels, format, chunk_duration=0.01):
    p = pyaudio.PyAudio()
    stream_out = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        output=True)

    chunk_size = int(rate * chunk_duration)
    total_chunks = len(audio_data) // chunk_size
    total_time = len(audio_data) / rate
    
    for i in range(total_chunks):
        segment = audio_data[i * chunk_size:(i + 1) * chunk_size]
        stream_out.write(segment.tobytes())
        
        amplitude, freq, energy = analyze_audio_segment(segment, rate)
        current_time = (i + 1) * chunk_duration
        
        print(f"Time: {current_time:.2f}/{total_time:.2f} sec | Amplitude: {amplitude:.2f}")
        time.sleep(chunk_duration)
    
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()

# Load a WAV file
def load_wav_file(file_path):
    with wave.open(file_path, 'rb') as wf:
        params = wf.getparams()
        channels = params.nchannels
        sample_width = params.sampwidth
        rate = params.framerate
        n_frames = params.nframes
        
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        
    return audio_data, rate, channels, sample_width

# Main testing function
def test_trim_silence(file_path, amp_threshold, frame_size, min_length=0.1):
    # Load the audio file
    audio_data, rate, channels, sample_width = load_wav_file(file_path)
    
    print("Playing original audio...")
    play_and_analyze_audio(audio_data, rate, channels, pyaudio.paInt16)
    
    # Measure the time before trimming
    start_time = time.time()
    
    # Trim silence
    trimmed_audio = trim_silence(audio_data, amp_threshold, frame_size, min_length)
    
    # Measure the time after trimming
    end_time = time.time()
    
    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Trimming took {duration:.4f} seconds")
    
    time.sleep(1)    

    print("Playing trimmed audio...")
    play_and_analyze_audio(trimmed_audio, rate, channels, pyaudio.paInt16)

# Function to calculate amplitude threshold
def calculate_amplitude_threshold(silence_directory, chunk_duration=0.01):
    """Calculate the threshold for silence based on the silence files."""
    silence_amplitude_values = []
    
    for filename in os.listdir(silence_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(silence_directory, filename)
            y, sr = librosa.load(filepath, sr=None)
            
            # Convert the floating-point values to the equivalent 16-bit integer range
            y = y * 32768  # Scale up to match int16 range
            
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

    # Check if the maximum amplitude is an extreme outlier compared to the average
    amp_multiplier = 1.5  # ::PARAMETER:: Adjust the multiplier as needed
    if max_amplitude > avg_amplitude * amp_multiplier:
        # Remove the extreme outlier and recalculate the max amplitude
        silence_amplitude_values = [amp for amp in silence_amplitude_values if amp < avg_amplitude * amp_multiplier]
        max_amplitude = np.max(silence_amplitude_values)

    # Set the silence threshold slightly above the maximum amplitude value of the (filtered) silence files
    return max_amplitude * 1.2  # ::PARAMETER:: Add a small margin to the calculated threshold

if __name__ == "__main__":
    # Replace 'your_audio_file.wav' with the name of your file
    file_path = 'data/recordings/Mic1/Mouse_Click/20240812_150503_recorded_output.wav'
    silence_directory = 'data/recordings/Mic1/Silence'
    amp_threshold = calculate_amplitude_threshold(silence_directory)
    print(f"Calculated Amplitude Threshold: {amp_threshold:.2f}")
    frame_size = 44100 // 100  # For 44100 Hz sampling rate and 0.02 sec frames
    
    test_trim_silence(file_path, amp_threshold, frame_size)
