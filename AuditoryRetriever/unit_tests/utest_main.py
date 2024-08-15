import unittest
from unittest.mock import patch, call, MagicMock, ANY
from src.main import Main

class TestMainScript(unittest.TestCase):

    @patch('src.main.keyboard')
    @patch('src.audio_processing.AudioProcessor')
    @patch('src.debugging.Debugger')
    def test_main(self, MockDebugger, MockAudioProcessor, mock_keyboard):
        mock_audio = MockAudioProcessor.return_value
        mock_dbg = MockDebugger.return_value

        mock_audio.process_audio.return_value = iter([(0, 1), (2, 3)])
        mock_dbg.toggle_debugging = MagicMock()
        mock_audio.toggle_audio = MagicMock()
        mock_audio.toggle_recording = MagicMock()
        mock_dbg.update_debugging_info = MagicMock()

        # Instantiate Main with mocked dependencies
        main = Main(audio_processor=mock_audio, debugger=mock_dbg)
        
        print("Before main function call")

        try:
            main.run()
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught")

        print("After main function call")

        try:
            mock_keyboard.on_press_key.assert_has_calls([
                call("p", ANY),
                call("o", ANY),
                call("r", ANY)
            ])
            print("Key event registrations were made correctly")
        except AssertionError as e:
            print(f"AssertionError: {e}")
        
        try:
            mock_keyboard.on_press_key.assert_has_calls([
                call("F13", ANY),
                call("F14", ANY),
                call("F15", ANY)
            ])
            print("Error: Unlikely key event registrations were found (this should not happen)")
        except AssertionError as e:
            print("Assertion passed for unlikely keys not being registered")
    
        mock_dbg.update_debugging_info.assert_has_calls([call(0, 1), call(2, 3)])
        print("update_debugging_info calls were made correctly")

if __name__ == '__main__':
    unittest.main()
