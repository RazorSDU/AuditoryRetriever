# debugging.py

import psutil
import os
import panel as pn

pn.extension()

class Debugger:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory_info = self.process.memory_info()
        self.initial_memory_used = self.initial_memory_info.rss / (1024 ** 2)  # Convert to MB
        self.cpu_usage_smoothed = 0
        self.debugging_enabled = False
        self.debug_info = ""

    def toggle_debugging(self):
        self.debugging_enabled = not self.debugging_enabled

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def update_debugging_info(self):
        if self.debugging_enabled:
            memory_info = self.process.memory_info()
            memory_used = memory_info.rss / (1024 ** 2)
            self.audio_memory_used = memory_used - self.initial_memory_used
            current_cpu_usage = self.get_cpu_usage()
            self.cpu_usage_smoothed = (self.cpu_usage_smoothed * 0.9) + (current_cpu_usage * 0.1)

            self.debug_info = (
                f"**Memory used by audio streams:** {self.audio_memory_used:.2f} MB | "
                f"**CPU usage:** {self.cpu_usage_smoothed:.2f}%"
            )
