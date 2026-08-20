import tempfile
import unittest
from pathlib import Path

from run_multilingual_pilot import parse_prediction


class PilotRunnerTests(unittest.TestCase):
    def test_prediction_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.txt'
            path.write_text(
                'A\t0-A,0.1\t1-B,0.2\n'
                'A\t1-B,0.1\t0-A,0.2\n'
                'B\t1-B,0.1\t0-A,0.2\n'
            )
            result = parse_prediction(path)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertEqual(result['per_label_accuracy'], {'A': 0.5, 'B': 1.0})


if __name__ == '__main__':
    unittest.main()
