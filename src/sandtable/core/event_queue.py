"""
src/sandtable/core/event_queue.py

Priority event queue for the event-driven backtesting framework.

Uses a min-heap to process events in timestamp order.
A counter ensures FIFO ordering for events with identical timestamps.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any

from sandtable.core.events import (
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    SignalEvent,
)
from sandtable.utils.logger import get_logger

logger = get_logger(__name__)

Event = MarketDataEvent | SignalEvent | OrderEvent | FillEvent


@dataclass
class _QueueEntry:
    """
    Internal wrapper for heap entries with proper comparison.

    Attributes:
        timestamp: Event timestamp for primary ordering
        counter: Insertion counter for FIFO tiebreaking
        event: The actual event object
    """

    timestamp: datetime
    counter: int
    event: Event

    def __lt__(self, other: _QueueEntry) -> bool:
        """
        Compare by timestamp first, then by counter for FIFO.
        """
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        return self.counter < other.counter

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _QueueEntry):
            return NotImplemented
        return self.timestamp == other.timestamp and self.counter == other.counter


class EventQueue:
    """
    Priority queue for processing events in temporal order.

    Events are processed in timestamp order. For events with the same
    timestamp, FIFO ordering is maintained using an internal counter.

    Example:
        >>> queue = EventQueue()
        >>> queue.push(market_event)
        >>> queue.push(signal_event)
        >>> while not queue.is_empty():
        ...     event = queue.pop()
        ...     process(event)
    """

    ## Magic methods

    def __init__(self) -> None:
        """
        Initialize an empty event queue.
        """
        self._heap: list[_QueueEntry] = []
        self._counter = count()
        logger.debug("EventQueue initialized")

    def __len__(self) -> int:
        """
        Return the number of events in the queue.
        """
        return len(self._heap)

    def __bool__(self) -> bool:
        """
        Return True if the queue has events.
        """
        return not self.is_empty()

    ## Public methods

    def push(self, event: Event) -> None:
        """
        Add an event to the queue.

        Args:
            event: Event to add (must have a timestamp attribute)
        """
        entry = _QueueEntry(
            timestamp=event.timestamp,
            counter=next(self._counter),
            event=event,
        )
        heapq.heappush(self._heap, entry)
        logger.debug(
            "Pushed %s event at %s (queue size: %d)",
            event.event_type.name,
            event.timestamp,
            len(self._heap),
        )

    def pop(self) -> Event:
        """
        Remove and return the next event by timestamp.

        Returns:
            The event with the earliest timestamp.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot pop from an empty EventQueue")
        entry = heapq.heappop(self._heap)
        logger.debug(
            "Popped %s event at %s (queue size: %d)",
            entry.event.event_type.name,
            entry.timestamp,
            len(self._heap),
        )
        return entry.event

    def peek(self) -> Event:
        """
        Return the next event without removing it.

        Returns:
            The event with the earliest timestamp.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek an empty EventQueue")
        return self._heap[0].event

    def is_empty(self) -> bool:
        """
        Check if the queue has no events.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return len(self._heap) == 0
