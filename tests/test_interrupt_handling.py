import io
import signal
import unittest
from contextlib import redirect_stderr

import podcast_rag.runtime as runtime


class FlushTrackingStream(io.StringIO):
    def __init__(self):
        super().__init__()
        self.was_flushed = False

    def flush(self):
        self.was_flushed = True
        super().flush()


class InterruptHandlingTests(unittest.TestCase):
    def test_ctrl_c_is_acknowledged_immediately_and_only_sets_stop_state(self):
        previous = runtime.STOP_REQUESTED
        previous_signal = runtime.STOP_SIGNAL
        stream = FlushTrackingStream()
        try:
            runtime.STOP_REQUESTED = False
            runtime.STOP_SIGNAL = None
            with redirect_stderr(stream):
                runtime.request_stop(signal.SIGINT, None)
            self.assertTrue(runtime.STOP_REQUESTED)
            self.assertEqual(signal.SIGINT, runtime.STOP_SIGNAL)
            self.assertTrue(stream.was_flushed)
            self.assertIn("Ctrl+C received", stream.getvalue())
            self.assertIn("before exiting", stream.getvalue())
        finally:
            runtime.STOP_REQUESTED = previous
            runtime.STOP_SIGNAL = previous_signal


if __name__ == "__main__":
    unittest.main()
