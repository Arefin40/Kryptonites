"""Regression checks for XROR decoding and vectorized quaternion processing."""

import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bson
import fpzip
import h5py
import numpy as np

from headpose import preprocess
from headpose.preprocess import TarPreprocessor


# ============================================================
# Small deterministic motion fixtures
# ============================================================


def rotations(times: np.ndarray, amplitude: float = 30.0) -> np.ndarray:
    angle = np.radians(amplitude) * np.sin(2 * np.pi * 0.5 * times)
    zeros = np.zeros(len(times))

    return np.column_stack((np.cos(angle / 2), zeros, np.sin(angle / 2), zeros))


def packed_replay(
    times: np.ndarray,
    quaternions: np.ndarray,
    *,
    events: list | None = None,
    head: bool = True,
) -> bytes:
    frames = np.column_stack((times, quaternions)).astype(np.float32)

    return bson.encode(
        {
            "info": {
                "hardware": {
                    "devices": [
                        {
                            "joint": "HEAD" if head else "HAND_LEFT",
                            "axes": ["w_rot", "x_rot", "y_rot", "z_rot"],
                        }
                    ]
                }
            },
            "frames": fpzip.compress(frames),
            "events": events or [],
        }
    )


# ============================================================
# Decoding and filter decisions
# ============================================================


class ReplayTests(unittest.TestCase):
    def test_unused_event_stream_is_not_decompressed(self):
        times = np.arange(100) / 90
        raw = packed_replay(
            times, rotations(times), events=[{"floatData": b"unused broken event data"}]
        )
        with patch.object(fpzip, "decompress", wraps=fpzip.decompress) as decoder:
            decoded = TarPreprocessor.read_xror(raw)

        self.assertEqual(decoder.call_count, 1)
        self.assertEqual(decoded[2:], (100, 0))
        np.testing.assert_allclose(np.linalg.norm(decoded[1], axis=1), 1.0)

    def test_invalid_and_duplicate_samples_keep_first_valid_row(self):
        times = np.array([0.02, 0.0, 0.01, 0.01, 0.03, np.nan, 0.04])
        q = rotations(np.nan_to_num(times))
        q[3] = [0, 1, 0, 0]
        q[4] = 0
        q[6] = np.nan
        raw = packed_replay(times, q)
        tt, qq, count, invalid = TarPreprocessor.read_xror(raw)

        self.assertEqual((count, invalid), (7, 4))
        np.testing.assert_allclose(tt, [0.0, 0.01, 0.02], atol=1e-8)
        np.testing.assert_allclose(qq[1], q[2], atol=1e-7)

    def test_missing_head_is_rejected_before_decompression(self):
        times = np.arange(100) / 90
        raw = packed_replay(times, rotations(times), head=False)
        with patch.object(fpzip, "decompress", wraps=fpzip.decompress) as decoder:
            with self.assertRaisesRegex(ValueError, "no device marked HEAD or HMD"):
                TarPreprocessor.read_xror(raw)
        self.assertEqual(decoder.call_count, 0)

    def test_all_invalid_quaternions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no valid head quaternion"):
            TarPreprocessor.read_xror(packed_replay(np.arange(10), np.zeros((10, 4))))

    def test_rate_rejection_and_inclusive_motion_thresholds(self):
        times = np.arange(3600) / 90
        q = rotations(times)
        result = TarPreprocessor.assess_session(times, q)
        self.assertEqual(result[4], "retained")
        with patch.object(preprocess, "SWEEP_MIN_DEG", result[1]):
            with patch.object(preprocess, "VELOCITY_MIN_DEG_S", result[2]):
                self.assertEqual(TarPreprocessor.assess_session(times, q)[4], "retained")
        with patch.object(preprocess, "SWEEP_MIN_DEG", np.nextafter(result[1], np.inf)):
            self.assertEqual(
                TarPreprocessor.assess_session(times, q)[4],
                "sustained sweep below 17 degrees",
            )
        times = np.arange(3600) / 72
        self.assertEqual(
            TarPreprocessor.assess_session(times, rotations(times))[4],
            "sampling rate outside 85-110 Hz",
        )


# ============================================================
# Numerical equivalence and output compatibility
# ============================================================


class QuaternionTests(unittest.TestCase):
    def test_alignment_resets_at_orthogonal_pairs(self):
        q = np.array(
            [
                [1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0],
                [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 1, 0],
            ],
            dtype=float,
        )
        original = q.copy()
        expected = q.copy()
        expected[1] *= -1
        expected[3] *= -1
        expected[5] *= -1
        np.testing.assert_array_equal(TarPreprocessor._align_hemispheres(q), expected)
        np.testing.assert_array_equal(q, original)

    def test_batch_slerp_matches_scalar_at_both_branches(self):
        rng = np.random.default_rng(92)
        first = rng.normal(size=(256, 4))
        second = rng.normal(size=(256, 4))
        first /= np.linalg.norm(first, axis=1, keepdims=True)
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        first[:4] = [1, 0, 0, 0]
        dots = np.array([1.0, -1.0, 0.99949, 0.99951])
        second[:4] = np.column_stack((dots, np.sqrt(1 - dots**2), np.zeros((4, 2))))
        fractions = rng.random(256)
        fractions[4:6] = [0, 1]
        expected = np.array(
            [TarPreprocessor._slerp(a, b, f) for a, b, f in zip(first, second, fractions)]
        )
        actual = TarPreprocessor._slerp_batch(first, second, fractions)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)

    def test_resampling_keeps_grid_endpoints_and_phase(self):
        times = np.array([0.003, 0.01, 0.021, 0.03, 0.049999999])
        q = rotations(times)
        q[2:] *= -1
        grid, actual = TarPreprocessor._resample_segment(times, q, 100)
        np.testing.assert_array_equal(grid, np.arange(0.01, 0.050000009, 0.01))
        q = TarPreprocessor._align_hemispheres(q)
        expected = []
        for target in grid:
            right = np.searchsorted(times, target)
            if right == len(times):
                expected.append(q[-1])
            elif times[right] == target or right == 0:
                expected.append(q[right])
            else:
                fraction = (target - times[right - 1]) / (times[right] - times[right - 1])
                expected.append(TarPreprocessor._slerp(q[right - 1], q[right], fraction))
        np.testing.assert_allclose(actual, np.array(expected, dtype=np.float32), atol=1e-7)

    def test_segment_length_floor_remains_420(self):
        processor = TarPreprocessor()
        for count, expected in [(419, 0), (420, 1)]:
            with self.subTest(count=count):
                times = np.arange(count) / 100
                self.assertEqual(len(processor.clean_segments(times, rotations(times))), expected)

    def test_mixed_archive_hdf5_and_manifest(self):
        times = np.arange(3600) / 90
        replays = {
            "moving": packed_replay(times, rotations(times)),
            "still": packed_replay(times, rotations(times, 0)),
            "broken": b"invalid BSON",
        }
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive_path = root / "sample.tar"
            with tarfile.open(archive_path, "w") as archive:
                for name, raw in replays.items():
                    member = tarfile.TarInfo(f"user/{name}.xror")
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
            output = root / "nested" / "sample.h5"
            results = TarPreprocessor().process_tar(archive_path, output, user_id="actual-user")
            self.assertEqual([r.kept for r in results], [True, False, False])
            TarPreprocessor.write_manifest(results, output.with_suffix(".csv"))
            with h5py.File(output, "r") as h5:
                self.assertEqual(list(h5["sessions"]), ["moving"])
                session = h5["sessions/moving"]
                self.assertEqual(session.attrs["user_id"], "actual-user")
                segment = session["segment_0000"]
                self.assertEqual(segment["quaternions_wxyz"].dtype, np.dtype("float32"))
                np.testing.assert_allclose(np.diff(segment["timestamps_s"][:]), 0.01)
                np.testing.assert_allclose(
                    np.linalg.norm(segment["quaternions_wxyz"][:], axis=1),
                    1.0, atol=1e-6,
                )


if __name__ == "__main__":
    unittest.main()
