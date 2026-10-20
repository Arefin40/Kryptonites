"""Filter one BOXRR-23 user archive and write clean head motion to HDF5."""

from __future__ import annotations

import argparse
import csv
import math
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

# ============================================================
# Challenge settings and result types
# ============================================================

RATE_MIN_HZ = 85.0
RATE_MAX_HZ = 110.0
SWEEP_MIN_DEG = 17.0
VELOCITY_MIN_DEG_S = 15.0
TRIM_SECONDS = 5.0
FILTER_WINDOW_SECONDS = 5.0
MIN_WINDOW_SAMPLES = 10
MIN_POST_TRIM_SAMPLES = 20
MIN_TRAINING_SAMPLES = 420
OUTPUT_RATE_HZ = 100.0
MIN_QUATERNION_NORM = 1e-8


@dataclass
class ReplayResult:
    replay: str
    sample_count: int = 0
    valid_sample_count: int = 0
    invalid_sample_count: int = 0
    sampling_rate_hz: float | None = None
    window_count: int = 0
    sweep_deg: float | None = None
    velocity_deg_s: float | None = None
    kept: bool = False
    output_segments: int = 0
    reason: str = ""


# ============================================================
# Preprocessing class
# ============================================================


class TarPreprocessor:
    """Process all XROR replays in one user tarball into a clean HDF5 file."""

    def __init__(
        self,
        *,
        max_gap_seconds: float = 0.1,
        output_rate_hz: float = OUTPUT_RATE_HZ,
    ) -> None:
        if max_gap_seconds <= 0.0:
            raise ValueError("max_gap_seconds must be positive")
        if output_rate_hz <= 0.0:
            raise ValueError("output_rate_hz must be positive")

        self.max_gap_seconds = max_gap_seconds
        self.output_rate_hz = output_rate_hz

    # ============================================================
    # XROR reading and quaternion conversion
    # ============================================================

    @staticmethod
    def _head_quaternion_indices(devices: list[dict[str, Any]]) -> tuple[int, ...]:
        """Return frame-column indices for head rotation in (w,x,y,z) order."""

        offset = 1  # Frame column zero is elapsed time.

        for device in devices:
            axes = [str(axis).strip().lower() for axis in device.get("axes", [])]
            is_head = (
                str(device.get("joint", "")).upper() == "HEAD"
                or str(device.get("type", "")).upper() == "HMD"
            )

            if is_head:
                axis_orders = (
                    (("i", "j", "k", "1"), ("1", "i", "j", "k")),
                    (
                        ("x_rot", "y_rot", "z_rot", "w_rot"),
                        ("w_rot", "x_rot", "y_rot", "z_rot"),
                    ),
                    (
                        ("rot_x", "rot_y", "rot_z", "rot_w"),
                        ("rot_w", "rot_x", "rot_y", "rot_z"),
                    ),
                )

                for available_names, challenge_order in axis_orders:
                    if all(name in axes for name in available_names):
                        return tuple(offset + axes.index(name) for name in challenge_order)

                raise ValueError("head/HMD device has no recognized quaternion axes")

            offset += len(axes)

        raise ValueError("XROR replay has no device marked HEAD or HMD")

    @classmethod
    def read_xror(cls, data: bytes) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Decode XROR bytes into seconds and normalized (w,x,y,z) quaternions."""

        try:
            from xror import XROR
        except ImportError as error:
            raise RuntimeError(
                "Install the project's XROR dependency to read .xror files"
            ) from error

        # ----------------------------------------
        # Decode the replay and locate head rotation columns
        # ----------------------------------------

        recording = XROR.unpack(data).data
        devices = recording.get("info", {}).get("hardware", {}).get("devices", [])
        frames = recording.get("frames", [])
        if not devices or not frames:
            raise ValueError("XROR replay has no devices or motion frames")

        indices = cls._head_quaternion_indices(devices)

        # ----------------------------------------
        # Validate and normalize each sample
        # ----------------------------------------

        timestamps: list[float] = []
        quaternions: list[tuple[float, float, float, float]] = []
        invalid = 0

        for frame in frames:
            try:
                timestamp = float(frame[0])
                quaternion = np.asarray(
                    [float(frame[index]) for index in indices], dtype=np.float64
                )
                norm = float(np.linalg.norm(quaternion))

                if (
                    not math.isfinite(timestamp)
                    or not np.isfinite(quaternion).all()
                    or norm < MIN_QUATERNION_NORM
                ):
                    raise ValueError("invalid timestamp or quaternion")

                timestamps.append(timestamp)
                quaternions.append(tuple((quaternion / norm).tolist()))
            except (TypeError, ValueError, IndexError, OverflowError):
                invalid += 1

        if not timestamps:
            raise ValueError("XROR replay has no valid head quaternion samples")

        # ----------------------------------------
        # Sort by time and remove duplicate timestamps
        # ----------------------------------------

        order = np.argsort(np.asarray(timestamps), kind="stable")
        times = np.asarray(timestamps, dtype=np.float64)[order]
        values = np.asarray(quaternions, dtype=np.float64)[order]
        unique = np.concatenate(([True], np.diff(times) > 0.0))
        invalid += int((~unique).sum())

        return times[unique], values[unique], len(frames), invalid

    @staticmethod
    def _to_euler_xyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
        """Convert normalized scalar-first quaternions to xyz angles in degrees."""

        w, x, y, z = quaternions_wxyz.T

        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return np.degrees(np.column_stack((roll, pitch, yaw)))

    # ============================================================
    # Challenge motion filters
    # ============================================================

    @staticmethod
    def _percentile(values: np.ndarray, quantile: float) -> float:
        """Return a percentile using linear interpolation."""

        return float(np.percentile(values, quantile, method="linear"))

    @staticmethod
    def _wrapped_velocity(values: np.ndarray, duration: float) -> float:
        """Measure angular path velocity while wrapping changes at 180 degrees."""

        if len(values) < 2 or duration <= 0.0:
            return 0.0

        differences = np.abs(np.diff(values))
        differences = np.where(differences > 180.0, 360.0 - differences, differences)

        return float(differences.sum() / duration)

    @classmethod
    def assess_session(
        cls,
        times: np.ndarray,
        quaternions_wxyz: np.ndarray,
    ) -> tuple[float | None, float | None, float | None, int, str, np.ndarray, np.ndarray]:
        """Apply sampling-rate and sustained-motion selection rules."""

        # ----------------------------------------
        # Estimate sampling rate
        # ----------------------------------------

        if len(times) < 2:
            return (
                None,
                None,
                None,
                0,
                "fewer than two valid timestamped samples",
                times,
                quaternions_wxyz,
            )

        intervals = np.diff(times)
        positive = intervals[np.isfinite(intervals) & (intervals > 0.0)]

        if not len(positive):
            return (
                None,
                None,
                None,
                0,
                "no positive inter-sample intervals",
                times,
                quaternions_wxyz,
            )

        rate = 1.0 / float(np.median(positive))

        if not RATE_MIN_HZ <= rate <= RATE_MAX_HZ:
            return (
                rate,
                None,
                None,
                0,
                "sampling rate outside 85-110 Hz",
                times,
                quaternions_wxyz,
            )

        # ----------------------------------------
        # Trim startup and validate remaining samples
        # ----------------------------------------

        start = times[0]
        keep = times - start >= TRIM_SECONDS
        trimmed_times = times[keep] - (start + TRIM_SECONDS)
        trimmed_quaternions = quaternions_wxyz[keep]

        if len(trimmed_times) < MIN_POST_TRIM_SAMPLES:
            return (
                rate,
                None,
                None,
                0,
                "fewer than 20 samples after the 5-second trim",
                trimmed_times,
                trimmed_quaternions,
            )

        # ----------------------------------------
        # Score non-overlapping 5-second windows
        # ----------------------------------------

        angles = cls._to_euler_xyz(trimmed_quaternions)
        sweep_windows: list[list[float]] = [[], [], []]
        velocity_windows: list[list[float]] = [[], [], []]
        window_count = 0
        duration = float(trimmed_times[-1])
        window_start = 0.0

        while window_start < duration:
            window_end = min(window_start + FILTER_WINDOW_SECONDS, duration)
            mask = (trimmed_times >= window_start) & (
                trimmed_times < window_start + FILTER_WINDOW_SECONDS
            )

            if int(mask.sum()) >= MIN_WINDOW_SAMPLES:
                window_count += 1
                window_duration = max(window_end - window_start, 1.0 / rate)

                for axis in range(3):
                    values = angles[mask, axis]
                    sweep = cls._percentile(values, 90) - cls._percentile(values, 10)
                    velocity = cls._wrapped_velocity(values, window_duration)
                    sweep_windows[axis].append(sweep)
                    velocity_windows[axis].append(velocity)

            window_start += FILTER_WINDOW_SECONDS

        if not window_count:
            return (
                rate,
                None,
                None,
                0,
                "no 5-second window has at least 10 valid samples",
                trimmed_times,
                trimmed_quaternions,
            )

        # ----------------------------------------
        # Aggregate window scores and apply thresholds
        # ----------------------------------------

        axis_sweeps = [float(np.median(values)) for values in sweep_windows if values]
        axis_velocities = [float(np.median(values)) for values in velocity_windows if values]
        sweep = max(axis_sweeps)
        velocity = max(axis_velocities)

        # This project follows the challenge overview's inclusive thresholds.
        if sweep < SWEEP_MIN_DEG:
            reason = "sustained sweep below 17 degrees"
        elif velocity < VELOCITY_MIN_DEG_S:
            reason = "sustained angular velocity below 15 degrees/second"
        else:
            reason = "retained"

        return rate, sweep, velocity, window_count, reason, trimmed_times, trimmed_quaternions

    # ============================================================
    # Quaternion alignment and 100 Hz cleanup
    # ============================================================

    @staticmethod
    def _align_hemispheres(quaternions: np.ndarray) -> np.ndarray:
        """Flip quaternion signs so adjacent samples stay in the same hemisphere."""

        aligned = quaternions.copy()

        for index in range(1, len(aligned)):
            if float(np.dot(aligned[index - 1], aligned[index])) < 0.0:
                aligned[index] *= -1.0

        return aligned

    @staticmethod
    def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
        """Interpolate between two quaternion rotations."""

        dot = float(np.dot(first, second))

        if dot < 0.0:
            second = -second
            dot = -dot

        dot = min(1.0, max(-1.0, dot))

        if dot > 0.9995:
            result = first + fraction * (second - first)
            return result / np.linalg.norm(result)

        angle = math.acos(dot)
        denominator = math.sin(angle)
        result = (
            math.sin((1.0 - fraction) * angle) / denominator * first
            + math.sin(fraction * angle) / denominator * second
        )

        return result / np.linalg.norm(result)

    @classmethod
    def _resample_segment(
        cls,
        times: np.ndarray,
        quaternions: np.ndarray,
        rate_hz: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate one uninterrupted segment onto a fixed-rate grid."""

        # ----------------------------------------
        # Build a uniform timestamp grid
        # ----------------------------------------

        quaternions = cls._align_hemispheres(quaternions)
        step = 1.0 / rate_hz
        first_grid_index = math.ceil(times[0] / step - 1e-12)
        grid = np.arange(
            first_grid_index * step,
            times[-1] + step * 1e-6,
            step,
            dtype=np.float64,
        )

        if not len(grid):
            return np.empty(0, dtype=np.float64), np.empty((0, 4), dtype=np.float32)

        # ----------------------------------------
        # Interpolate orientation for each grid point
        # ----------------------------------------

        output = np.empty((len(grid), 4), dtype=np.float32)

        for index, target in enumerate(grid):
            right = int(np.searchsorted(times, target, side="left"))

            if right == 0:
                output[index] = quaternions[0]
            elif right >= len(times):
                output[index] = quaternions[-1]
            elif times[right] == target:
                output[index] = quaternions[right]
            else:
                left = right - 1
                fraction = float((target - times[left]) / (times[right] - times[left]))
                output[index] = cls._slerp(quaternions[left], quaternions[right], fraction)

        return grid, output

    def clean_segments(
        self,
        times: np.ndarray,
        quaternions_wxyz: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split long dropouts, interpolate short gaps, and resample each segment."""

        # ----------------------------------------
        # Split at dropouts that exceed the configured limit
        # ----------------------------------------

        if len(times) < 2:
            return []

        breaks = np.flatnonzero(np.diff(times) > self.max_gap_seconds) + 1
        time_parts = np.split(times, breaks)
        quaternion_parts = np.split(quaternions_wxyz, breaks)

        # ----------------------------------------
        # Resample segments and enforce the 420-sample floor
        # ----------------------------------------

        segments: list[tuple[np.ndarray, np.ndarray]] = []

        for part_times, part_quaternions in zip(time_parts, quaternion_parts):
            if len(part_times) < 2:
                continue

            segment_times, segment_quaternions = self._resample_segment(
                part_times,
                part_quaternions,
                self.output_rate_hz,
            )

            if len(segment_quaternions) >= MIN_TRAINING_SAMPLES:
                segments.append((segment_times, segment_quaternions))

        return segments

    # ============================================================
    # HDF5 output and replay manifest
    # ============================================================

    @staticmethod
    def write_manifest(results: Iterable[ReplayResult], output: Path) -> None:
        """Write one processing result per replay to CSV."""

        output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(ReplayResult.__dataclass_fields__)

        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()

            for result in results:
                writer.writerow(asdict(result))

    def process_tar(
        self,
        archive_path: Path,
        output_path: Path,
        *,
        user_id: str | None = None,
    ) -> list[ReplayResult]:
        """Process each XROR replay in a user tarball and write retained segments."""

        # ----------------------------------------
        # Prepare output and open the source archive
        # ----------------------------------------

        results: list[ReplayResult] = []
        archive_user = user_id or archive_path.stem
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, mode="r:*") as archive:
            replay_members = [
                member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).suffix.lower() == ".xror"
            ]

            if not replay_members:
                raise ValueError(f"No .xror replay files found in {archive_path}")

            with h5py.File(output_path, "w") as h5:
                h5.attrs["source_archive"] = archive_path.name
                h5.attrs["quaternion_order"] = "w,x,y,z"
                h5.attrs["sampling_rate_hz"] = self.output_rate_hz
                h5.attrs["sweep_threshold_deg"] = SWEEP_MIN_DEG
                h5.attrs["velocity_threshold_deg_s"] = VELOCITY_MIN_DEG_S
                sessions_group = h5.create_group("sessions")

                # ----------------------------------------
                # Process each replay independently
                # ----------------------------------------

                for member in replay_members:
                    result = self._process_replay(
                        archive,
                        member,
                        sessions_group,
                        archive_user,
                        len(results),
                    )
                    results.append(result)

        return results

    def _process_replay(
        self,
        archive: tarfile.TarFile,
        member: tarfile.TarInfo,
        sessions_group: h5py.Group,
        user_id: str,
        replay_index: int,
    ) -> ReplayResult:
        """Read, filter, clean, and optionally write one replay."""

        replay_name = Path(member.name).stem
        result = ReplayResult(replay=replay_name)

        try:
            # ----------------------------------------
            # Read XROR motion and apply challenge filters
            # ----------------------------------------

            source = archive.extractfile(member)

            if source is None:
                raise ValueError("could not read replay from tar archive")

            with source:
                raw = source.read()

            times, quaternions, count, invalid = self.read_xror(raw)
            result.sample_count = count
            result.valid_sample_count = len(times)
            result.invalid_sample_count = invalid
            (
                result.sampling_rate_hz,
                result.sweep_deg,
                result.velocity_deg_s,
                result.window_count,
                result.reason,
                trimmed_times,
                trimmed_quaternions,
            ) = self.assess_session(times, quaternions)

            result.kept = result.reason == "retained"

            if not result.kept:
                return result

            # ----------------------------------------
            # Clean retained data and prepare HDF5 groups
            # ----------------------------------------

            segments = self.clean_segments(trimmed_times, trimmed_quaternions)

            if not segments:
                result.kept = False
                result.reason = "no segment has at least 420 samples after cleaning"
                return result

            replay_key = replay_name

            if replay_key in sessions_group:
                replay_key = f"{replay_name}_{replay_index:05d}"

            replay_group = sessions_group.create_group(replay_key)
            replay_group.attrs["user_id"] = user_id
            replay_group.attrs["source_member"] = member.name
            replay_group.attrs["estimated_source_rate_hz"] = result.sampling_rate_hz
            replay_group.attrs["sampling_rate_tag_hz"] = self.output_rate_hz
            replay_group.attrs["sweep_deg"] = result.sweep_deg
            replay_group.attrs["velocity_deg_s"] = result.velocity_deg_s

            # ----------------------------------------
            # Write each continuous segment
            # ----------------------------------------

            for segment_index, (segment_times, segment_quaternions) in enumerate(segments):
                segment_group = replay_group.create_group(f"segment_{segment_index:04d}")
                segment_group.create_dataset(
                    "timestamps_s",
                    data=segment_times,
                    compression="lzf",
                    shuffle=True,
                )
                segment_group.create_dataset(
                    "quaternions_wxyz",
                    data=segment_quaternions,
                    compression="lzf",
                    shuffle=True,
                )

            result.output_segments = len(segments)

        except Exception as error:
            result.kept = False
            result.reason = f"processing error: {error}"

        return result


# ============================================================
# Command-line interface
# ============================================================


def main() -> int:
    """Run the tarball preprocessing command."""

    # ----------------------------------------
    # Define command arguments
    # ----------------------------------------

    parser = argparse.ArgumentParser(
        description="Filter and preprocess the XROR replays in one BOXRR-23 user tarball."
    )
    parser.add_argument("archive", type=Path, help="Path to one user's .tar archive")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HDF5 file (default: outputs/preprocessed/<archive>.h5)",
    )
    parser.add_argument("--manifest", type=Path, help="Optional replay decision CSV")
    parser.add_argument(
        "--user-id",
        help="User identifier to store in HDF5; defaults to archive filename",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=0.1,
        help="Largest dropout to fill with SLERP; longer gaps split the session (default: 0.1)",
    )
    parser.add_argument(
        "--output-rate-hz",
        type=float,
        default=OUTPUT_RATE_HZ,
        help="Clean output rate (default: 100)",
    )

    # ----------------------------------------
    # Validate arguments
    # ----------------------------------------

    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")
    if args.max_gap_seconds <= 0.0 or args.output_rate_hz <= 0.0:
        parser.error("gap limit and output rate must be positive")

    # ----------------------------------------
    # Process archive and write manifest
    # ----------------------------------------

    output_path = args.output or Path("outputs/preprocessed") / f"{args.archive.stem}.h5"
    processor = TarPreprocessor(
        max_gap_seconds=args.max_gap_seconds,
        output_rate_hz=args.output_rate_hz,
    )
    results = processor.process_tar(
        args.archive,
        output_path,
        user_id=args.user_id,
    )

    manifest_path = args.manifest or output_path.with_suffix(".csv")
    processor.write_manifest(results, manifest_path)

    # ----------------------------------------
    # Print summary
    # ----------------------------------------

    retained = sum(result.kept for result in results)
    print(
        f"Processed {len(results)} replays; retained {retained}; "
        f"rejected {len(results) - retained}."
    )
    print(f"HDF5: {output_path}")
    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
