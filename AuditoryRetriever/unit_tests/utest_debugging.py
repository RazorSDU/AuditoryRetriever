import unittest
from unittest.mock import patch, MagicMock
from src.debugging import Debugger

class TestDebugging(unittest.TestCase):

    @patch('src.debugging.psutil.Process')
    def setUp(self, mock_process):
        self.debugger = Debugger()
        self.mock_process = mock_process.return_value
        self.mock_process.memory_info.return_value.rss = 100 * (1024 ** 2)  # 100 MB
        self.debugger.process = self.mock_process
        self.debugger.initial_memory_info = self.mock_process.memory_info()
        self.debugger.initial_memory_used = self.debugger.initial_memory_info.rss / (1024 ** 2)
    
    def tearDown(self):
        self.debugger.debugging_enabled = False

    def test_toggle_debugging(self):
        self.debugger.toggle_debugging()
        self.assertTrue(self.debugger.debugging_enabled)
        self.debugger.toggle_debugging()
        self.assertFalse(self.debugger.debugging_enabled)

    @patch('src.debugging.psutil.cpu_percent', return_value=50.0)
    def test_get_cpu_usage(self, mock_cpu_percent):
        cpu_usage = self.debugger.get_cpu_usage()
        self.assertEqual(cpu_usage, 50.0)
        
    @patch('src.debugging.psutil.cpu_percent', return_value=50.0)
    @patch('src.debugging.time.time', side_effect=[0, 1])
    @patch('builtins.print')
    def test_update_debugging_info(self, mock_print, mock_time, mock_cpu_percent):
        self.debugger.debugging_enabled = True
        self.debugger.update_debugging_info(0, 1)
        self.assertTrue(mock_print.called)

if __name__ == '__main__':
    unittest.main()
