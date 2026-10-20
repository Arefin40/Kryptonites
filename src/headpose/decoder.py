"""Isolate native XROR decoding failures from the preprocessing process."""

from __future__ import annotations

import multiprocessing
from typing import Any


# ============================================================
# Child process entry point
# ============================================================


def _decode_worker(connection: Any) -> None:
    from headpose.preprocess import TarPreprocessor

    try:
        while True:
            data = connection.recv()
            if data is None:
                return

            try:
                result = TarPreprocessor.read_xror(data)
            except Exception as error:
                connection.send((False, f"{type(error).__name__}: {error}"))
            else:
                connection.send((True, result))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


# ============================================================
# Persistent decoder with automatic recovery
# ============================================================


class ReplayDecoder:
    """Reuse one spawned decoder per archive; restart after a native crash."""

    def __init__(self, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._process: Any = None
        self._connection: Any = None

    def __enter__(self) -> ReplayDecoder:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the pipe and reap the child, including stalled decoders."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None

        if self._process is not None:
            self._process.join(timeout=1)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
            self._process = None

    def decode(self, data: bytes) -> Any:
        # ----------------------------------------
        # Spawn safely from downloader threads without inheriting HDF5 handles
        # ----------------------------------------

        if self._process is None:
            context = multiprocessing.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(target=_decode_worker, args=(child,), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            self._connection = parent
            self._process = process

        # ----------------------------------------
        # Native crashes become replay errors; the next call starts a new child
        # ----------------------------------------

        try:
            self._connection.send(data)
            if not self._connection.poll(self.timeout_seconds):
                raise TimeoutError(
                    f"XROR decoder exceeded {self.timeout_seconds:g} seconds"
                )
            success, result = self._connection.recv()
        except TimeoutError:
            self.close()
            raise
        except (EOFError, BrokenPipeError, OSError) as error:
            self._process.join(timeout=1)
            exit_code = self._process.exitcode
            self.close()
            raise RuntimeError(
                f"XROR decoder exited unexpectedly (exit code {exit_code}); "
                "replay skipped"
            ) from error
        except BaseException:
            self.close()
            raise

        if not success:
            raise ValueError(result)
        return result
