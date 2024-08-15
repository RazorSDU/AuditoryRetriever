# main.py


import sys
import os
import time
import panel as pn
import threading
from src.audio_processing import AudioProcessor
from src.debugging import Debugger

# Add the root directory of the project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pn.extension()

def get_audio_processor():
    return AudioProcessor()

def get_debugger():
    return Debugger()

class Main:
    def __init__(self, audio_processor=None, debugger=None):
        self.audio_processor = audio_processor or get_audio_processor()
        self.debugger = debugger or get_debugger()
        self.prev_states = {
            'audio_enabled': None,
            'recording': None,
            'debugging': None
        }

        # Panel widgets
        self.icon_button = pn.widgets.Button(name="", button_type="primary", width=40)
        self.recording_toggle_button = pn.widgets.Toggle(name="Start Recording", button_type="primary")
        self.audio_toggle_button = pn.widgets.Toggle(name="Enable Audio", button_type="success")
        self.debugging_toggle_button = pn.widgets.Toggle(name="Enable Debugging", button_type="warning")

        # Buttons for Sound Recognizer options
        self.fft_button = pn.widgets.Toggle(name="FFT", button_type="primary")
        self.mfcc_button = pn.widgets.Toggle(name="MFCC", button_type="primary")
        self.spectral_button = pn.widgets.Toggle(name="Spectral Contrast", button_type="primary")
        self.chroma_button = pn.widgets.Toggle(name="Chroma", button_type="primary")
        self.zero_crossing_button = pn.widgets.Toggle(name="Zero Crossing", button_type="primary")
        self.cnn_button = pn.widgets.Toggle(name="CNN", button_type="primary")

        # Panel placeholders for displaying individual scores
        self.fft_score_clap = pn.pane.Markdown("")
        self.fft_score_desk_tap = pn.pane.Markdown("")
        self.fft_score_keyboard_space = pn.pane.Markdown("")
        self.fft_score_mouse_click = pn.pane.Markdown("")
        self.fft_score_silence = pn.pane.Markdown("")
        self.fft_score_snap = pn.pane.Markdown("")

        self.mfcc_score_clap = pn.pane.Markdown("")
        self.mfcc_score_desk_tap = pn.pane.Markdown("")
        self.mfcc_score_keyboard_space = pn.pane.Markdown("")
        self.mfcc_score_mouse_click = pn.pane.Markdown("")
        self.mfcc_score_silence = pn.pane.Markdown("")
        self.mfcc_score_snap = pn.pane.Markdown("")

        self.spectral_score_clap = pn.pane.Markdown("")
        self.spectral_score_desk_tap = pn.pane.Markdown("")
        self.spectral_score_keyboard_space = pn.pane.Markdown("")
        self.spectral_score_mouse_click = pn.pane.Markdown("")
        self.spectral_score_silence = pn.pane.Markdown("")
        self.spectral_score_snap = pn.pane.Markdown("")

        self.chroma_score_clap = pn.pane.Markdown("")
        self.chroma_score_desk_tap = pn.pane.Markdown("")
        self.chroma_score_keyboard_space = pn.pane.Markdown("")
        self.chroma_score_mouse_click = pn.pane.Markdown("")
        self.chroma_score_silence = pn.pane.Markdown("")
        self.chroma_score_snap = pn.pane.Markdown("")

        self.zero_crossing_score_clap = pn.pane.Markdown("")
        self.zero_crossing_score_desk_tap = pn.pane.Markdown("")
        self.zero_crossing_score_keyboard_space = pn.pane.Markdown("")
        self.zero_crossing_score_mouse_click = pn.pane.Markdown("")
        self.zero_crossing_score_silence = pn.pane.Markdown("")
        self.zero_crossing_score_snap = pn.pane.Markdown("")

        self.cnn_score_clap = pn.pane.Markdown("")
        self.cnn_score_desk_tap = pn.pane.Markdown("")
        self.cnn_score_keyboard_space = pn.pane.Markdown("")
        self.cnn_score_mouse_click = pn.pane.Markdown("")
        self.cnn_score_silence = pn.pane.Markdown("")
        self.cnn_score_snap = pn.pane.Markdown("")

        # Link widgets to functions
        self.recording_toggle_button.param.watch(self.toggle_recording, 'value')
        self.audio_toggle_button.param.watch(self.toggle_audio, 'value')
        self.debugging_toggle_button.param.watch(self.toggle_debugging, 'value')

        self.fft_button.param.watch(self.toggle_fft, 'value')
        self.mfcc_button.param.watch(self.toggle_mfcc, 'value')
        self.spectral_button.param.watch(self.toggle_spectral, 'value')
        self.chroma_button.param.watch(self.toggle_chroma, 'value')
        self.zero_crossing_button.param.watch(self.toggle_zero_crossing, 'value')
        self.cnn_button.param.watch(self.toggle_cnn, 'value')

    def toggle_audio(self, event):
        self.audio_processor.toggle_audio(event.new)
        self.prev_states['audio_enabled'] = event.new
        self.audio_toggle_button.name = "Disable Audio" if event.new else "Enable Audio"
        self.audio_toggle_button.button_type = "danger" if event.new else "success"

    def toggle_recording(self, event):
        if event.new:
            self.audio_processor.start_recording()
            self.recording_toggle_button.name = "Stop Recording"
            self.recording_toggle_button.button_type = "danger"
        else:
            self.audio_processor.stop_recording()
            self.recording_toggle_button.name = "Start Recording"
            self.recording_toggle_button.button_type = "primary"

    def toggle_debugging(self, event):
        self.debugger.toggle_debugging()
        self.prev_states['debugging'] = event.new
        self.debugging_toggle_button.name = "Disable Debugging" if event.new else "Enable Debugging"
        self.debugging_toggle_button.button_type = "danger" if event.new else "warning"

    def toggle_fft(self, event):
        self.audio_processor.set_use_fft(event.new)
        self.fft_button.button_type = "danger" if event.new else "primary"

    def toggle_mfcc(self, event):
        self.audio_processor.set_use_mfcc(event.new)
        self.mfcc_button.button_type = "danger" if event.new else "primary"
    
    def toggle_cnn(self, event):
        self.audio_processor.set_use_cnn(event.new)
        self.cnn_button.button_type = "danger" if event.new else "primary"

    def toggle_spectral(self, event):
        self.audio_processor.set_use_spectral_contrast(event.new)
        self.spectral_button.button_type = "danger" if event.new else "primary"

    def toggle_chroma(self, event):
        self.audio_processor.set_use_chroma_features(event.new)
        self.chroma_button.button_type = "danger" if event.new else "primary"

    def toggle_zero_crossing(self, event):
        self.audio_processor.set_use_zero_crossing(event.new)
        self.zero_crossing_button.button_type = "danger" if event.new else "primary"

    def update_recognition_scores(self):
        recognition_scores = self.audio_processor.recognition_result_placeholder

        def initialize_score_dict(score_dict, expected_keys):
            # Ensure all expected keys are present with a default value of 0 if missing
            return {key: score_dict.get(key, 0) for key in expected_keys}

        def set_color(score_dict):
            scores = list(score_dict.values())
            # If there are duplicates for the highest score, all should be white
            if scores.count(max(scores)) > 1:
                return {key: "white" for key in score_dict}
            else:
                max_score = max(scores)
                return {key: ("green" if score == max_score else "white") for key, score in score_dict.items()}

        def update_scores_and_colors(score_dict, pane_objects, expected_keys):
            score_dict = initialize_score_dict(score_dict, expected_keys)
            colors = set_color(score_dict)
            for sound_type, pane_object in pane_objects.items():
                score = score_dict.get(sound_type, 0)
                color = colors[sound_type]
                pane_object.object = f'<span style="color:{color}">{sound_type.replace("_", " ").title()}: {score:.2f}%</span>'

        fft_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        fft_scores = recognition_scores.get('fft', {})
        update_scores_and_colors(
            fft_scores,
            {
                'clap': self.fft_score_clap,
                'desk_tap': self.fft_score_desk_tap,
                'keyboard_space': self.fft_score_keyboard_space,
                'mouse_click': self.fft_score_mouse_click,
                'silence': self.fft_score_silence,
                'snap': self.fft_score_snap,
            },
            fft_keys
        )

        mfcc_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        mfcc_scores = recognition_scores.get('mfcc', {})
        update_scores_and_colors(
            mfcc_scores,
            {
                'clap': self.mfcc_score_clap,
                'desk_tap': self.mfcc_score_desk_tap,
                'keyboard_space': self.mfcc_score_keyboard_space,
                'mouse_click': self.mfcc_score_mouse_click,
                'silence': self.mfcc_score_silence,
                'snap': self.mfcc_score_snap,
            },
            mfcc_keys
        )

        spectral_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        spectral_scores = recognition_scores.get('spectral_contrast', {})
        update_scores_and_colors(
            spectral_scores,
            {
                'clap': self.spectral_score_clap,
                'desk_tap': self.spectral_score_desk_tap,
                'keyboard_space': self.spectral_score_keyboard_space,
                'mouse_click': self.spectral_score_mouse_click,
                'silence': self.spectral_score_silence,
                'snap': self.spectral_score_snap,
            },
            spectral_keys
        )

        chroma_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        chroma_scores = recognition_scores.get('chroma', {})
        update_scores_and_colors(
            chroma_scores,
            {
                'clap': self.chroma_score_clap,
                'desk_tap': self.chroma_score_desk_tap,
                'keyboard_space': self.chroma_score_keyboard_space,
                'mouse_click': self.chroma_score_mouse_click,
                'silence': self.chroma_score_silence,
                'snap': self.chroma_score_snap,
            },
            chroma_keys
        )

        zero_crossing_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        zero_crossing_scores = recognition_scores.get('zero_crossing', {})
        update_scores_and_colors(
            zero_crossing_scores,
            {
                'clap': self.zero_crossing_score_clap,
                'desk_tap': self.zero_crossing_score_desk_tap,
                'keyboard_space': self.zero_crossing_score_keyboard_space,
                'mouse_click': self.zero_crossing_score_mouse_click,
                'silence': self.zero_crossing_score_silence,
                'snap': self.zero_crossing_score_snap,
            },
            zero_crossing_keys
        )

        cnn_keys = ['clap', 'desk_tap', 'keyboard_space', 'mouse_click', 'silence', 'snap']
        cnn_scores = recognition_scores.get('cnn', {})
        update_scores_and_colors(
            cnn_scores,
            {
                'clap': self.cnn_score_clap,
                'desk_tap': self.cnn_score_desk_tap,
                'keyboard_space': self.cnn_score_keyboard_space,
                'mouse_click': self.cnn_score_mouse_click,
                'silence': self.cnn_score_silence,
                'snap': self.cnn_score_snap,
            },
            cnn_keys
        )


    def update_debug_info(self):
        self.debugger.update_debugging_info()
            

    def run(self):
        # Create a row for the three buttons
        button_row = pn.Row(
            self.recording_toggle_button,
            self.audio_toggle_button,
            self.debugging_toggle_button,
            sizing_mode="stretch_width"
        )

        # Create columns for each recognition type with their respective buttons and scores
        recognition_FFT_Column = pn.Column(
            self.fft_button,
            self.fft_score_clap,
            self.fft_score_desk_tap,
            self.fft_score_keyboard_space,
            self.fft_score_mouse_click,
            self.fft_score_silence,
            self.fft_score_snap,
            sizing_mode="stretch_width"
        )

        recognition_MFCC_Column = pn.Column(
            self.mfcc_button,
            self.mfcc_score_clap,
            self.mfcc_score_desk_tap,
            self.mfcc_score_keyboard_space,
            self.mfcc_score_mouse_click,
            self.mfcc_score_silence,
            self.mfcc_score_snap,
            sizing_mode="stretch_width"
        )

        recognition_Spectral_Column = pn.Column(
            self.spectral_button,
            self.spectral_score_clap,
            self.spectral_score_desk_tap,
            self.spectral_score_keyboard_space,
            self.spectral_score_mouse_click,
            self.spectral_score_silence,
            self.spectral_score_snap,
            sizing_mode="stretch_width"
        )

        recognition_Chroma_Column = pn.Column(
            self.chroma_button,
            self.chroma_score_clap,
            self.chroma_score_desk_tap,
            self.chroma_score_keyboard_space,
            self.chroma_score_mouse_click,
            self.chroma_score_silence,
            self.chroma_score_snap,
            sizing_mode="stretch_width"
        )

        recognition_Zero_Crossing_Column = pn.Column(
            self.zero_crossing_button,
            self.zero_crossing_score_clap,
            self.zero_crossing_score_desk_tap,
            self.zero_crossing_score_keyboard_space,
            self.zero_crossing_score_mouse_click,
            self.zero_crossing_score_silence,
            self.zero_crossing_score_snap,
            sizing_mode="stretch_width"
        )

        recognition_CNN_Column = pn.Column(
            self.cnn_button,
            self.cnn_score_clap,
            self.cnn_score_desk_tap,
            self.cnn_score_keyboard_space,
            self.cnn_score_mouse_click,
            self.cnn_score_silence,
            self.cnn_score_snap,
            sizing_mode="stretch_width"
        )

        # Create a row for the recognition columns
        recognition_row = pn.Row(
            recognition_FFT_Column,
            recognition_MFCC_Column,
            recognition_Spectral_Column,
            recognition_Chroma_Column,
            recognition_Zero_Crossing_Column,
            recognition_CNN_Column,
            sizing_mode="stretch_width"
        )

        # Create the layout
        layout = pn.Column(
            "# Auditory Retriever",
            pn.Row(
                self.recording_toggle_button,
                self.audio_toggle_button,
                self.debugging_toggle_button,
                sizing_mode="stretch_width"
            ),
            recognition_row,
            sizing_mode="stretch_width"
        )

        # Start the audio processing in a separate thread
        audio_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        audio_thread.start()

        # Use a periodic callback to update the UI
        pn.state.add_periodic_callback(self.update_recognition_scores, period=100)
        pn.state.add_periodic_callback(self.update_debug_info, period=100)

        # Show the app
        pn.serve(layout, show=True, start=True)

    def process_audio_stream(self):
        try:
            while True:
                if self.audio_processor.audio_enabled or self.audio_processor.recording or self.debugger.debugging_enabled:
                    self.audio_processor.process_audio()
                time.sleep(0.1)  # Sleep for a short while
        except KeyboardInterrupt:
            print("Stopping the playback")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main = Main()
    main.run()
