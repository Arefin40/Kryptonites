"""Build a resumable HDF5 corpus from an already downloaded prefix folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from headpose.downloader import LOG_DIR, OUTPUT_DIR, RemoteUserPreprocessor
from headpose.preprocess import ReplayResult, TarPreprocessor


# ============================================================
# Local archive inventory and processing
# ============================================================


@dataclass(frozen=True)
class LocalArchive:
    path: str
    local_path: Path
    size: int
    blob_id: str


class LocalFolderPreprocessor(RemoteUserPreprocessor):
    """Reuse the bounded workers, serial HDF5 merge, manifest and resume logic."""

    def __init__(
        self,
        input_dir: Path,
        *,
        max_workers: int = 2,
        min_training_samples: int = 1000,
        max_gap_seconds: float = 0.1,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        prefix = self.input_dir.name
        if len(prefix) != 1 or prefix not in "0123456789abcdef":
            raise ValueError("Input directory must be a single prefix folder: 0-9 or a-f")
        super().__init__(repo_id="local", max_workers=max_workers)
        self.user_folder = f"users/{prefix}"
        self.processor = TarPreprocessor(
            max_gap_seconds=max_gap_seconds,
            min_training_samples=min_training_samples,
            strict_motion_thresholds=True,
        )
        self.config = {
            "pipeline_version": 1,
            "source_directory": str(self.input_dir),
            "min_training_samples": min_training_samples,
            "max_gap_seconds": max_gap_seconds,
            "output_rate_hz": 100.0,
            "strict_motion_thresholds": True,
        }

    def _migrate_previous_outputs(self, **kwargs) -> None:
        """Local runs never import legacy remote output files."""

    def _list_archives(self, user_folder: str) -> tuple[str, list[LocalArchive]]:
        entries = []
        config_bytes = json.dumps(self.config, sort_keys=True).encode()
        for path in sorted(self.input_dir.iterdir()):
            if not path.is_file() or not self._is_archive(path.name):
                continue
            user_id = self._archive_user_id(path.name)
            if not user_id.startswith(self.input_dir.name):
                raise ValueError(f"Archive user ID does not match prefix: {path.name}")
            stat = path.stat()
            fingerprint = hashlib.sha256(
                config_bytes + f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            entries.append(LocalArchive(
                f"{user_folder}/{path.name}", path, stat.st_size, fingerprint
            ))
        return "local-v1", entries

    def _download_and_process(
        self, *, entry: LocalArchive, resolved_revision: str,
        user_id: str, output_dir: Path,
    ) -> tuple[Path, list[ReplayResult]]:
        # ----------------------------------------
        # Read the original TAR directly; only temporary HDF5 needs extra disk
        # ----------------------------------------

        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=".headpose-local-", dir=output_dir))
        try:
            self.logger.info("Preprocessing local archive %s", entry.local_path)
            results = self.processor.process_tar(
                entry.local_path, temporary_root / "archive.h5", user_id=user_id
            )
            stat = entry.local_path.stat()
            fingerprint = hashlib.sha256(
                json.dumps(self.config, sort_keys=True).encode()
                + f"{entry.local_path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            if fingerprint != entry.blob_id:
                raise RuntimeError("Source archive changed during processing; retry it")
            return temporary_root, results
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise

    def run(self, output_dir: Path = OUTPUT_DIR, *, log_dir: Path = LOG_DIR):
        # ----------------------------------------
        # Prevent mixing outputs from remote runs or different filtering policies
        # ----------------------------------------

        shard = f"user-{self.input_dir.name}"
        destination = output_dir.expanduser().resolve() / shard
        destination.mkdir(parents=True, exist_ok=True)
        config_path = destination / "local_pipeline.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != self.config:
                raise ValueError("Local pipeline configuration changed; use a new output directory")
        elif any(destination.glob(f"{shard}.*")) or any(destination.glob(".*preprocess-state.json")):
            raise ValueError("Existing output has no local configuration; use a new output directory")
        else:
            temporary = config_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.config, indent=2) + "\n")
            temporary.replace(config_path)
        return self.process_folder(self.user_folder, output_dir, log_dir=log_dir)


# ============================================================
# Command-line interface
# ============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Downloaded prefix folder, e.g. data/raw/users/d")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR / "local")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-training-samples", type=int, default=1000)
    parser.add_argument("--max-gap-seconds", type=float, default=0.1)
    args = parser.parse_args()
    try:
        runner = LocalFolderPreprocessor(
            args.input_dir, max_workers=args.workers,
            min_training_samples=args.min_training_samples,
            max_gap_seconds=args.max_gap_seconds,
        )
        _, _, _, _, failed = runner.run(args.output_dir, log_dir=args.log_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
