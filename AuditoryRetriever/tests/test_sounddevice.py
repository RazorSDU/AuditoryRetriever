import sounddevice as sd
import numpy as np
import time as time_module  # Renamed to avoid conflict with callback argument
import psutil
import os
import sys

# Parameters
samplerate = 44100  # Sample rate in Hz
blocksize = 1024    # Block size in samples
channels = 1        # Mono

# Initialize variables for delay calculation
delay = 0

# Get initial memory usage before starting the audio stream
process = psutil.Process(os.getpid())
initial_memory_info = process.memory_info()
initial_memory_used = initial_memory_info.rss / (1024 ** 2)  # Convert to MB

# Callback function to process audio blocks
def callback(indata, outdata, frames, time, status):
    global delay
    if status:
        print(status)
    
    # Playback the recorded audio
    outdata[:] = indata
    
    # Calculate the delay for the current block
    current_delay = time.outputBufferDacTime - time.currentTime
    
    # Smooth out the delay calculation to avoid abrupt changes
    delay = (delay * 0.9) + (current_delay * 0.1)

# Start the audio stream
with sd.Stream(samplerate=samplerate, blocksize=blocksize, channels=channels, callback=callback):
    print("Press Ctrl+C to stop the playback")
    try:
        while True:
            # Get the current memory usage
            memory_info = process.memory_info()
            memory_used = memory_info.rss / (1024 ** 2)  # Convert to MB
            
            # Calculate the memory usage by the audio stream
            audio_memory_used = memory_used - initial_memory_used
            
            # Print delay and memory usage on the same line
            sys.stdout.write(f"\rDelay: {delay:.6f} seconds | Memory used by audio stream: {audio_memory_used:.2f} MB")
            sys.stdout.flush()
            
            # Sleep for a short duration to reduce CPU usage
            time_module.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping the playback")
