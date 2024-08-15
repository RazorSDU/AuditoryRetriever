import unittest
from unittest.mock import patch, MagicMock, call
import os
import numpy as np
from datetime import datetime
from src.audio_processing import AudioProcessor

class TestAudioProcessing(unittest.TestCase):

    @patch('src.audio_processing.pyaudio.PyAudio')
    def setUp(self, MockPyAudio):
        # Mock the Debugger instance
        self.mock_debugger = MagicMock()
        # Initialize AudioProcessor with the mocked Debugger
        self.audio_processor = AudioProcessor(debugger=self.mock_debugger)
        self.mock_p = MockPyAudio.return_value
        self.mock_stream = self.mock_p.open.return_value
        self.mock_stream.read.return_value = b'\x00\x01' * 1024  # Mocked audio data
        self.mock_stream_out = self.mock_p.open.return_value
        self.audio_processor.stream = self.mock_stream
        self.audio_processor.stream_out = self.mock_stream_out
        self.audio_processor.p = self.mock_p
        self.audio_processor.record_buffer = MagicMock()
        self.audio_processor.print_recording_info = MagicMock()

    def test_toggle_audio(self):
        # Test toggling the audio state
        initial_state = self.audio_processor.audio_enabled
        
        self.audio_processor.toggle_audio()
        self.assertNotEqual(self.audio_processor.audio_enabled, initial_state)

        self.audio_processor.toggle_audio()
        self.assertEqual(self.audio_processor.audio_enabled, initial_state)

    @patch('src.audio_processing.os.makedirs')
    @patch('src.audio_processing.sf.write')
    @patch('src.audio_processing.datetime')
    def test_save_audio(self, mock_datetime, mock_sf_write, mock_os_makedirs):
        # Test saving audio data to a file
        mock_now = MagicMock()
        mock_now.strftime.return_value = '20240527_120000'
        mock_datetime.now.return_value = mock_now
        
        data = b'\x00\x01' * 1024
        filename = 'test.wav'
        self.audio_processor.save_audio(data, filename)
        
        mock_os_makedirs.assert_called_once_with('data/recordings', exist_ok=True)
        expected_file_path = os.path.join('data/recordings', '20240527_120000_test.wav')
        data_np = np.frombuffer(data, dtype=np.int16)
        
        mock_sf_write.assert_called_once()
        call_args = mock_sf_write.call_args
        self.assertEqual(call_args[0][0], expected_file_path)
        np.testing.assert_array_equal(call_args[0][1], data_np)
        self.assertEqual(call_args[0][2], 44100)

    @patch('src.audio_processing.AudioProcessor.save_audio')
    def test_toggle_recording(self, mock_save_audio):
        # Test toggling the recording state
        initial_recording_state = self.audio_processor.recording
        self.audio_processor.toggle_recording()
        self.assertNotEqual(self.audio_processor.recording, initial_recording_state)
        
        self.audio_processor.toggle_recording()
        self.assertFalse(self.audio_processor.recording)
        mock_save_audio.assert_called_once()

if __name__ == '__main__':
    unittest.main()
