"""Run slow output writes on a background thread.

Compressing head logits is pure host work, so it need not hold up the next
seed. :class:`BackgroundWriter` runs those writes on one worker thread, in
submission order. The caller keeps three guarantees:

* A failed write fails the run: the next :meth:`~BackgroundWriter.submit` or
  :meth:`~BackgroundWriter.drain` raises, and leaving the ``with`` block drains.
* The ``with`` block does not exit, normally or on an exception, until every
  submitted write has finished.
* At most ``max_pending`` writes are queued, so a slow disk blocks the producer
  instead of piling logits up in host memory.

Only submit work whose bytes do not depend on the device. Anything computed on
the GPU (confidence metrics) stays on the caller's thread, because the same
arithmetic on CPU may round differently.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepfold3.model.model import DeepFold3Output

logger = logging.getLogger(__name__)


class BackgroundWriter:
    """One worker thread running output writes in submission order.

    Args:
        max_pending: Most writes queued or running at once; ``submit`` blocks
            beyond that.
    """

    def __init__(self, max_pending: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepfold3-writer")
        self._slots = threading.Semaphore(max_pending)
        self._futures: list[Future] = []
        self._failures: list[tuple[str, BaseException]] = []

    def _run(self, label: str, fn: Callable[..., Any], args: tuple, kwargs: dict) -> None:
        try:
            fn(*args, **kwargs)
        except BaseException as e:  # noqa: B902 - MemoryError and friends must reach the caller too
            self._failures.append((label, e))
        finally:
            self._slots.release()

    def _raise_failures(self) -> None:
        if self._failures:
            label, first = self._failures[0]
            raise RuntimeError(f"{len(self._failures)} background write(s) failed, first: {label}: {first}") from first

    def submit(self, label: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> None:
        """Queue ``fn(*args, **kwargs)``. ``label`` names the output in errors.

        The arguments must be host data the caller will not modify.

        Raises:
            RuntimeError: An earlier write failed.
        """
        self._raise_failures()
        self._slots.acquire()
        self._futures = [f for f in self._futures if not f.done()]
        self._futures.append(self._pool.submit(self._run, label, fn, args, kwargs))

    def drain(self) -> None:
        """Wait for every submitted write.

        Raises:
            RuntimeError: Any write failed. Failures stay recorded, so a later
                ``drain`` raises again.
        """
        wait(self._futures)
        self._futures.clear()
        self._raise_failures()

    def __enter__(self) -> BackgroundWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.drain()
        except RuntimeError:
            if exc is None:
                raise
            # The body already failed and that exception propagates; name the
            # lost write here rather than replace it.
            logger.exception("background write failed while handling %r", exc)
        finally:
            self._pool.shutdown(wait=True)


def logits_to_host(output: DeepFold3Output) -> DeepFold3Output:
    """Copy of ``output`` with PAE, PDE and distogram logits on the host.

    One bulk transfer on the caller's thread, so the worker never touches the
    GPU. The other fields stay as they are; the logit savers do not read them.
    """

    def host(x):
        return None if x is None else x.detach().float().cpu()

    return dataclasses.replace(output, pae=host(output.pae), pde=host(output.pde), distogram=host(output.distogram))
