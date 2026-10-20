"""Download user archives from one prefix folder and preprocess them."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from tqdm import tqdm

from headpose.preprocess import OUTPUT_RATE_HZ, ReplayResult, TarPreprocessor

# ============================================================
# Dataset selection
# ============================================================

DATASET_REPO_ID = "cschell/boxrr-23"
DATASET_REVISION = "main"
USER_FOLDER = "users/d"
OUTPUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")
MAX_WORKERS = 2
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


# ============================================================
# Logging setup
# ============================================================


class TqdmLoggingHandler(logging.Handler):
    """Write log records without breaking the active tqdm progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def configure_logger(log_dir: Path, user_folder: str) -> logging.Logger:
    """Create console and persistent file logs for the selected folder."""

    log_dir.mkdir(parents=True, exist_ok=True)
    folder_name = Path(user_folder.strip("/")).name or "dataset"
    log_path = log_dir / f"{folder_name}.log"
    logger = logging.getLogger("headpose.downloader")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Log file: %s", log_path.resolve())

    return logger


# ============================================================
# Remote user-folder processing
# ============================================================


class RemoteUserPreprocessor:
    """Fetch and process a remote folder with bounded concurrent workers."""

    ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")

    def __init__(
        self,
        *,
        repo_id: str = DATASET_REPO_ID,
        revision: str = DATASET_REVISION,
        max_gap_seconds: float = 0.1,
        output_rate_hz: float = OUTPUT_RATE_HZ,
        max_workers: int = MAX_WORKERS,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        # Hide Hub download/reconstruction bars; keep our overall tqdm progress.
        disable_progress_bars()

        self.repo_id = repo_id
        self.revision = revision
        self.max_workers = max_workers
        self.token = HF_TOKEN
        self.logger = logging.getLogger("headpose.downloader")
        self.processor = TarPreprocessor(
            max_gap_seconds=max_gap_seconds,
            output_rate_hz=output_rate_hz,
        )

    @classmethod
    def _is_archive(cls, path: str) -> bool:
        return path.lower().endswith(cls.ARCHIVE_SUFFIXES)

    @staticmethod
    def _archive_key(remote_path: str) -> str:
        digest = hashlib.sha256(remote_path.encode("utf-8")).hexdigest()[:20]

        return f"archive_{digest}"

    @classmethod
    def _archive_user_id(cls, remote_path: str) -> str:
        """Read the user ID from a BOXRR-23 user-tar filename."""

        filename = Path(remote_path).name

        for suffix in sorted(cls.ARCHIVE_SUFFIXES, key=len, reverse=True):
            if filename.lower().endswith(suffix):
                return filename[: -len(suffix)]

        raise ValueError(f"unsupported user archive path: {remote_path}")

    @staticmethod
    def _archive_group_path(user_id: str, archive_key: str) -> str:
        return f"users/{user_id}/archives/{archive_key}"

    @staticmethod
    def _read_state(state_path: Path) -> dict[str, Any]:
        """Load previously completed remote-file signatures."""

        if not state_path.exists():
            return {"archives": {}}

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"archives": {}}

        if not isinstance(state.get("archives"), dict):
            return {"archives": {}}

        return state

    @staticmethod
    def _write_state(state_path: Path, state: dict[str, Any]) -> None:
        """Write checkpoint through a temporary file to avoid partial JSON."""

        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary_path.replace(state_path)

    @staticmethod
    def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
        if not manifest_path.exists():
            return []

        with manifest_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def _write_manifest(
        manifest_path: Path,
        rows: list[dict[str, Any]],
    ) -> None:
        """Atomically rewrite the combined replay report."""

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        fieldnames = ["source_tar", "user_id", *ReplayResult.__dataclass_fields__]

        with temporary_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        temporary_path.replace(manifest_path)

    def _migrate_previous_outputs(
        self,
        *,
        output_root: Path,
        user_output_dir: Path,
        folder_name: str,
    ) -> None:
        """Rename older shard outputs while preserving their checkpoints."""

        legacy_root = Path("outputs/preprocessed").resolve()

        # ----------------------------------------
        # Rename files already stored in this shard directory
        # ----------------------------------------

        for extension in ("h5", "csv"):
            source_path = user_output_dir / f"{folder_name}.{extension}"
            destination_path = user_output_dir / f"user-{folder_name}.{extension}"

            if source_path.exists() and not destination_path.exists():
                source_path.replace(destination_path)
                self.logger.info("Renamed previous shard output: %s", destination_path)

        old_state_path = user_output_dir / f".{folder_name}.preprocess-state.json"
        new_state_path = user_output_dir / f".user-{folder_name}.preprocess-state.json"

        if old_state_path.exists() and not new_state_path.exists():
            old_state_path.replace(new_state_path)
            self.logger.info("Renamed previous shard checkpoint: %s", new_state_path)

        if output_root != OUTPUT_DIR.resolve() or not legacy_root.is_dir():
            return

        # ----------------------------------------
        # Keep outputs from the original flat layout resumable
        # ----------------------------------------

        for extension in ("h5", "csv"):
            source_path = legacy_root / f"{folder_name}.{extension}"
            destination_path = user_output_dir / f"user-{folder_name}.{extension}"

            if source_path.exists() and not destination_path.exists():
                source_path.replace(destination_path)
                self.logger.info("Moved previous shard output: %s", destination_path)

        old_state_path = legacy_root / f".{folder_name}.preprocess-state.json"
        new_state_path = user_output_dir / f".user-{folder_name}.preprocess-state.json"

        if old_state_path.exists() and not new_state_path.exists():
            old_state_path.replace(new_state_path)
            self.logger.info("Moved previous shard checkpoint: %s", new_state_path)

    def _migrate_archive_groups(self, h5_path: Path, user_folder: str) -> None:
        """Nest each archive under its actual user ID and repair old metadata."""

        if not h5_path.exists():
            return

        with h5py.File(h5_path, "a") as h5:
            h5.attrs["user_folder"] = user_folder

            if "user_id" in h5.attrs:
                del h5.attrs["user_id"]

            # ----------------------------------------
            # Move groups from the old flat archive layout
            # ----------------------------------------

            if "archives" in h5:
                for archive_key in list(h5["archives"].keys()):
                    archive_group = h5[f"archives/{archive_key}"]
                    source_tar = str(archive_group.attrs.get("source_tar", ""))

                    try:
                        user_id = self._archive_user_id(source_tar)
                    except ValueError:
                        self.logger.warning(
                            "Cannot recover user ID for archive group %s", archive_key
                        )
                        continue

                    destination_path = self._archive_group_path(user_id, archive_key)
                    h5.require_group(f"users/{user_id}/archives")
                    if destination_path not in h5:
                        h5.move(f"archives/{archive_key}", destination_path)

            # ----------------------------------------
            # Ensure all stored replay metadata uses the actual user ID
            # ----------------------------------------

            if "users" not in h5:
                return

            for user_id in h5["users"]:
                archive_root = h5[f"users/{user_id}/archives"]

                for archive_key in archive_root:
                    archive_group = archive_root[archive_key]
                    archive_group.attrs["user_id"] = user_id

                    if "sessions" not in archive_group:
                        continue

                    for replay_group in archive_group["sessions"].values():
                        replay_group.attrs["user_id"] = user_id

    def _list_archives(self, user_folder: str) -> tuple[str, list[Any]]:
        """Resolve the remote revision and list only tar files in its folder."""

        api = HfApi(token=self.token)
        repo_info = api.repo_info(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=self.revision,
        )
        resolved_revision = repo_info.sha
        entries = api.list_repo_tree(
            repo_id=self.repo_id,
            repo_type="dataset",
            revision=resolved_revision,
            path_in_repo=user_folder.strip("/"),
            recursive=True,
        )
        archives = sorted(
            (entry for entry in entries if self._is_archive(entry.path)),
            key=lambda entry: entry.path.lower(),
        )

        return resolved_revision, archives

    @staticmethod
    def _signature(entry: Any) -> dict[str, Any]:
        """Use Hub blob identity and size to detect changed remote archives."""

        return {
            "blob_id": getattr(entry, "blob_id", None),
            "size": getattr(entry, "size", None),
        }

    def _download_and_process(
        self,
        *,
        entry: Any,
        resolved_revision: str,
        user_id: str,
        output_dir: Path,
    ) -> tuple[Path, list[ReplayResult]]:
        """Download and process one archive in a private temporary directory."""

        # ----------------------------------------
        # Download one archive into an isolated temporary cache
        # ----------------------------------------

        self.logger.info("Downloading %s", entry.path)
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=".headpose-download-",
                dir=output_dir,
            )
        )

        try:
            local_archive = Path(
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=entry.path,
                    repo_type="dataset",
                    revision=resolved_revision,
                    cache_dir=temporary_root / "hub-cache",
                    token=self.token,
                )
            )
            temporary_h5 = temporary_root / "archive.h5"

            self.logger.info("Preprocessing %s", entry.path)
            worker_processor = TarPreprocessor(
                max_gap_seconds=self.processor.max_gap_seconds,
                output_rate_hz=self.processor.output_rate_hz,
            )
            results = worker_processor.process_tar(
                local_archive,
                temporary_h5,
                user_id=user_id,
            )

        except Exception:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise

        return temporary_root, results

    def _merge_archive_output(
        self,
        *,
        temporary_h5: Path,
        entry: Any,
        resolved_revision: str,
        user_id: str,
        output_path: Path,
        archive_key: str,
        signature: dict[str, Any],
    ) -> None:
        """Merge one completed temporary HDF5 into the shared file serially."""

        # ----------------------------------------
        # Append the completed archive after its worker has closed the file
        # ----------------------------------------

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(temporary_h5, "r") as source_h5:
            with h5py.File(output_path, "a") as destination_h5:
                destination_h5.attrs["user_folder"] = str(Path(entry.path).parent)

                if "user_id" in destination_h5.attrs:
                    del destination_h5.attrs["user_id"]

                destination_h5.attrs["source_repo"] = self.repo_id
                destination_h5.attrs["source_revision"] = resolved_revision
                destination_h5.attrs["quaternion_order"] = "w,x,y,z"
                destination_h5.attrs["sampling_rate_hz"] = self.processor.output_rate_hz
                archive_group_path = self._archive_group_path(user_id, archive_key)
                archive_group = destination_h5.require_group(archive_group_path)
                archive_group.attrs["user_id"] = user_id
                archive_group.attrs["source_tar"] = entry.path
                archive_group.attrs["source_blob_id"] = str(signature["blob_id"] or "")
                archive_group.attrs["source_size_bytes"] = int(signature["size"] or 0)
                archive_group.attrs["source_revision"] = resolved_revision
                source_h5.copy("sessions", archive_group, name="sessions")

    def process_folder(
        self,
        user_folder: str,
        output_dir: Path = OUTPUT_DIR,
        *,
        log_dir: Path = LOG_DIR,
    ) -> tuple[Path, Path, int, int, int]:
        """Process one remote folder, skipping archives already in the checkpoint."""

        # ----------------------------------------
        # Resolve remote files and prepare persistent outputs
        # ----------------------------------------

        user_folder = user_folder.strip("/")
        self.logger = configure_logger(log_dir, user_folder)

        if not user_folder:
            raise ValueError("user_folder must be a remote path such as users/0-f")

        folder_name = Path(user_folder).name
        shard_name = f"user-{folder_name}"
        output_root = output_dir.expanduser().resolve()
        user_output_dir = output_root / shard_name
        user_output_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_previous_outputs(
            output_root=output_root,
            user_output_dir=user_output_dir,
            folder_name=folder_name,
        )
        h5_path = user_output_dir / f"{shard_name}.h5"
        manifest_path = user_output_dir / f"{shard_name}.csv"
        state_path = user_output_dir / f".{shard_name}.preprocess-state.json"
        self._migrate_archive_groups(h5_path, user_folder)
        state = self._read_state(state_path)
        manifest_rows = self._read_manifest(manifest_path)

        # ----------------------------------------
        # Recover actual user IDs for older CSV rows
        # ----------------------------------------

        for row in manifest_rows:
            if row.get("user_id"):
                continue

            try:
                row["user_id"] = self._archive_user_id(row.get("source_tar", ""))
            except ValueError:
                row["user_id"] = ""

        resolved_revision, archives = self._list_archives(user_folder)

        if not archives:
            raise ValueError(
                f"no supported tar archives found in "
                f"{self.repo_id}/{user_folder} at {resolved_revision}"
            )

        self.logger.info(
            "Starting folder %s from %s at revision %s: %d tar files found",
            user_folder,
            self.repo_id,
            resolved_revision,
            len(archives),
        )

        processed = 0
        skipped = 0
        failed = 0
        pending: list[tuple[Any, str, dict[str, Any]]] = []

        # ----------------------------------------
        # Identify completed archives and clear interrupted partial outputs
        # ----------------------------------------

        progress = tqdm(
            total=len(archives),
            desc=f"{user_folder} archives",
            unit="tar",
            dynamic_ncols=True,
        )

        for entry in archives:
            archive_path = entry.path
            archive_key = self._archive_key(archive_path)
            archive_user_id = self._archive_user_id(archive_path)
            signature = self._signature(entry)

            if state["archives"].get(archive_path) == signature and h5_path.exists():
                with h5py.File(h5_path, "r") as h5:
                    archive_group_path = self._archive_group_path(
                        archive_user_id,
                        archive_key,
                    )
                    if archive_group_path in h5:
                        skipped += 1
                        self.logger.info("Skipping already completed archive %s", archive_path)
                        progress.update(1)
                        continue

            # ----------------------------------------
            # Discard any incomplete copy before retrying this tar
            # ----------------------------------------

            if h5_path.exists():
                with h5py.File(h5_path, "a") as h5:
                    archive_group_path = self._archive_group_path(
                        archive_user_id,
                        archive_key,
                    )
                    if archive_group_path in h5:
                        del h5[archive_group_path]

            pending.append((entry, archive_key, signature))

        pending_paths = {entry.path for entry, _, _ in pending}
        manifest_rows = [
            row for row in manifest_rows if row.get("source_tar") not in pending_paths
        ]
        self._write_manifest(manifest_path, manifest_rows)
        progress.set_postfix(done=processed, skipped=skipped, failed=failed)

        self.logger.info(
            "Processing %d pending archives with %d workers",
            len(pending),
            self.max_workers,
        )

        # ----------------------------------------
        # Download and preprocess a bounded number of archives concurrently
        # ----------------------------------------

        pending_iterator = iter(pending)
        futures: dict[Any, tuple[Any, str, dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            def submit_next() -> bool:
                try:
                    entry, archive_key, signature = next(pending_iterator)
                except StopIteration:
                    return False

                future = executor.submit(
                    self._download_and_process,
                    entry=entry,
                    resolved_revision=resolved_revision,
                    user_id=self._archive_user_id(entry.path),
                    output_dir=user_output_dir,
                )
                futures[future] = (entry, archive_key, signature)

                return True

            for _ in range(min(self.max_workers, len(pending))):
                submit_next()

            while futures:
                completed_futures, _ = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in completed_futures:
                    entry, archive_key, signature = futures.pop(future)
                    archive_user_id = self._archive_user_id(entry.path)
                    temporary_root: Path | None = None

                    try:
                        temporary_root, results = future.result()
                        self._merge_archive_output(
                            temporary_h5=temporary_root / "archive.h5",
                            entry=entry,
                            resolved_revision=resolved_revision,
                            user_id=archive_user_id,
                            output_path=h5_path,
                            archive_key=archive_key,
                            signature=signature,
                        )
                        manifest_rows.extend(
                            {
                                "source_tar": entry.path,
                                "user_id": archive_user_id,
                                **asdict(result),
                            }
                            for result in results
                        )
                        manifest_rows.sort(
                            key=lambda row: (row["source_tar"], row["replay"])
                        )
                        self._write_manifest(manifest_path, manifest_rows)
                        state["archives"][entry.path] = signature
                        self._write_state(state_path, state)
                        processed += 1
                        retained = sum(result.kept for result in results)
                        self.logger.info(
                            "Completed %s: %d replays, %d retained",
                            entry.path,
                            len(results),
                            retained,
                        )

                    except Exception as error:
                        failed += 1
                        self.logger.exception(
                            "Failed to process %s: %s",
                            entry.path,
                            error,
                        )

                    finally:
                        if temporary_root is not None:
                            shutil.rmtree(temporary_root, ignore_errors=True)

                        progress.update(1)
                        progress.set_postfix(
                            done=processed,
                            skipped=skipped,
                            failed=failed,
                        )
                        submit_next()

        progress.close()

        self.logger.info(
            "Finished folder %s: processed=%d skipped=%d failed=%d",
            user_folder,
            processed,
            skipped,
            failed,
        )

        return h5_path, manifest_path, processed, skipped, failed


# ============================================================
# Command-line interface
# ============================================================


def main() -> int:
    """Process the configured remote folder, with optional CLI overrides."""

    # ----------------------------------------
    # Define command arguments
    # ----------------------------------------

    parser = argparse.ArgumentParser(
        description="Download and preprocess tar archives in one Hugging Face folder."
    )
    parser.add_argument(
        "--user-folder",
        default=USER_FOLDER,
        help=f"Remote folder path in the dataset repo (default: {USER_FOLDER})",
    )
    parser.add_argument("--repo-id", default=DATASET_REPO_ID)
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--max-gap-seconds", type=float, default=0.1)
    parser.add_argument("--output-rate-hz", type=float, default=OUTPUT_RATE_HZ)
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Maximum archives to download and preprocess at once (default: {MAX_WORKERS})",
    )
    args = parser.parse_args()

    if args.max_gap_seconds <= 0.0 or args.output_rate_hz <= 0.0:
        parser.error("gap limit and output rate must be positive")
    if args.workers < 1:
        parser.error("workers must be at least 1")

    # ----------------------------------------
    # Process selected remote folder
    # ----------------------------------------

    runner = RemoteUserPreprocessor(
        repo_id=args.repo_id,
        revision=args.revision,
        max_gap_seconds=args.max_gap_seconds,
        output_rate_hz=args.output_rate_hz,
        max_workers=args.workers,
    )
    logger = logging.getLogger("headpose.downloader")

    try:
        h5_path, manifest_path, processed, skipped, failed = runner.process_folder(
            args.user_folder,
            args.output_dir,
            log_dir=args.log_dir,
        )
    except Exception:
        logger.exception("Could not process remote folder %s", args.user_folder)
        return 1

    logger.info(
        "Run summary: processed=%d skipped=%d failed=%d",
        processed,
        skipped,
        failed,
    )
    logger.info("HDF5: %s", h5_path)
    logger.info("Manifest: %s", manifest_path)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
