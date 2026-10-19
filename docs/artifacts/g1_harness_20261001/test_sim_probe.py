"""Run with the native checkout Python/PYTHONPATH; starts no processes."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ProbeArtifactsTest(unittest.TestCase):
    def test_existing_evidence_is_rejected_before_resources_open(self):
        spec = importlib.util.spec_from_file_location(
            "sonic_sim_probe", Path(__file__).with_name("sonic_sim_probe.py")
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / "events.jsonl").write_text("existing evidence\n")
            (output / "release-band").touch()
            argv = ["probe", "--native", "unused", "--profile", "unused", "--output", str(output)]
            with patch("sys.argv", argv), patch.object(
                probe.zmq, "Context", side_effect=AssertionError("Opened resources before validating output")
            ):
                with self.assertRaises(FileExistsError):
                    probe.main()
            self.assertEqual((output / "events.jsonl").read_text(), "existing evidence\n")


if __name__ == "__main__":
    unittest.main()
