import pyaudio
import time
import psutil
import os
import sys

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Mono
RATE = 44100              # Sample rate
CHUNK = 1024              # Block size

# Initialize PyAudio
p = pyaudio.PyAudio()

# Get initial memory usage before starting the audio stream
process = psutil.Process(os.getpid())
initial_memory_info = process.memory_info()
initial_memory_used = initial_memory_info.rss / (1024 ** 2)  # Convert to MB

# Open input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Open output stream
stream_out = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

# Initialize delay for smoothing
delay = 0

print("Press Ctrl+C to stop the playback")

try:
    while True:
        # Record the start time
        start_time = time.time()
        
        # Read data from input stream
        try:
            data = stream.read(CHUNK)
        except IOError as e:
            print(f"Stream read error: {e}")
            continue
        
        # Write data to output stream
        stream_out.write(data)
        
        # Record the end time
        end_time = time.time()
        
        # Calculate the current delay
        current_delay = end_time - start_time
        
        # Smooth out the delay calculation to avoid abrupt changes
        delay = (delay * 0.9) + (current_delay * 0.1)
        
        # Get the current memory usage
        memory_info = process.memory_info()
        memory_used = memory_info.rss / (1024 ** 2)  # Convert to MB
        
        # Calculate the memory usage by the audio streams
        audio_memory_used = memory_used - initial_memory_used
        
        # Print delay and memory usage on the same line
        sys.stdout.write(f"\rDelay: {delay:.6f} seconds | Memory used by audio streams: {audio_memory_used:.2f} MB")
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\nStopping the playback")
finally:
    # Stop and close streams
    stream.stop_stream()
    stream.close()
    stream_out.stop_stream()
    stream_out.close()
    # Terminate PyAudio
    p.terminate()
