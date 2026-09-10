"""Stopping long work that is already running.

A generation run is hours and a search is many short trainings, so "I have seen
enough, stop" is an ordinary thing to want. Python cannot kill a thread, so
stopping is cooperative: the caller supplies a predicate, the long loops ask it
at the points where stopping is *safe*, and the answer is honest about what
that means -- work already checkpointed is kept, and the step in flight
finishes first.

The alternative, killing a worker outright, would leave a half-written dataset
that the next run would have to distrust. Between "stops instantly and corrupts
state" and "stops at the next chain boundary", a spin-chain simulation wants
the second.
"""

from __future__ import annotations

import threading
from typing import Callable


class OperationCancelled(RuntimeError):
    """Raised inside a long operation when the caller has asked it to stop.

    A subclass of ``RuntimeError`` so an unprepared caller still sees an error
    rather than a silent partial result, and distinct so a prepared one can
    tell "the user stopped this" apart from "this broke".
    """


class CancelToken:
    """A thread-safe "should I stop?" that can be checked or asserted.

    Wraps :class:`threading.Event` rather than a bare flag so a waiter can
    block on it, and so setting it from the request thread is visible to the
    worker without a lock of its own.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def __call__(self) -> bool:
        """So a token can be passed anywhere a ``should_stop`` predicate goes."""
        return self._event.is_set()

    def check(self, what: str = "the operation") -> None:
        """Raise if cancellation has been requested; otherwise do nothing."""
        if self._event.is_set():
            raise OperationCancelled(f"{what} was stopped before it finished")

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


def check_cancelled(
    should_stop: Callable[[], bool] | None, what: str = "the operation"
) -> None:
    """Raise :class:`OperationCancelled` if ``should_stop`` says to stop.

    Takes ``None`` so every call site can pass its optional predicate straight
    through without guarding first, which is what keeps these checks from
    cluttering the loops they sit in.
    """
    if should_stop is not None and should_stop():
        raise OperationCancelled(f"{what} was stopped before it finished")


__all__ = ["CancelToken", "OperationCancelled", "check_cancelled"]
