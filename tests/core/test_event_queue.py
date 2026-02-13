"""tests/core/test_event_queue.py

Tests for EventQueue functionality.
"""

from datetime import datetime, timedelta

import pytest

from backtester.core.event_queue import EventQueue
from backtester.core.events import (
    Direction,
    MarketDataEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
)


def make_market_event(timestamp: datetime, symbol: str = "SPY") -> MarketDataEvent:
    """
    Helper to create a MarketDataEvent.
    """
    return MarketDataEvent(
        timestamp=timestamp,
        symbol=symbol,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1_000_000,
    )


def make_signal_event(timestamp: datetime, symbol: str = "SPY") -> SignalEvent:
    """
    Helper to create a SignalEvent.
    """
    return SignalEvent(
        timestamp=timestamp,
        symbol=symbol,
        direction=Direction.LONG,
        strength=1.0,
    )


def make_order_event(timestamp: datetime, symbol: str = "SPY") -> OrderEvent:
    """
    Helper to create an OrderEvent.
    """
    return OrderEvent(
        timestamp=timestamp,
        symbol=symbol,
        direction=Direction.LONG,
        quantity=100,
        order_type=OrderType.MARKET,
    )


class TestEventQueueBasics:
    """
    Test basic queue operations.
    """

    def test_empty_queue(self):
        """
        New queue should be empty.
        """
        queue = EventQueue()
        assert queue.is_empty()
        assert len(queue) == 0
        assert not queue  # bool conversion

    def test_push_makes_non_empty(self):
        """
        Pushing an event makes queue non-empty.
        """
        queue = EventQueue()
        event = make_market_event(datetime(2024, 1, 1))
        queue.push(event)

        assert not queue.is_empty()
        assert len(queue) == 1
        assert queue  # bool conversion

    def test_pop_returns_event(self):
        """
        Pop returns the pushed event.
        """
        queue = EventQueue()
        event = make_market_event(datetime(2024, 1, 1))
        queue.push(event)

        popped = queue.pop()
        assert popped == event
        assert queue.is_empty()

    def test_peek_returns_without_removing(self):
        """
        Peek returns event without removing it.
        """
        queue = EventQueue()
        event = make_market_event(datetime(2024, 1, 1))
        queue.push(event)

        peeked = queue.peek()
        assert peeked == event
        assert len(queue) == 1  # Still in queue


class TestEventQueueOrdering:
    """
    Test timestamp-based ordering.
    """

    def test_timestamp_ordering(self):
        """
        Events should be popped in timestamp order.
        """
        queue = EventQueue()
        base = datetime(2024, 1, 1)

        # Push in reverse order
        event3 = make_market_event(base + timedelta(days=2))
        event1 = make_market_event(base)
        event2 = make_market_event(base + timedelta(days=1))

        queue.push(event3)
        queue.push(event1)
        queue.push(event2)

        # Should pop in timestamp order
        assert queue.pop() == event1
        assert queue.pop() == event2
        assert queue.pop() == event3

    def test_fifo_for_same_timestamp(self):
        """
        Events with same timestamp should be FIFO.
        """
        queue = EventQueue()
        ts = datetime(2024, 1, 1)

        # Different event types, same timestamp
        market = make_market_event(ts)
        signal = make_signal_event(ts)
        order = make_order_event(ts)

        # Push in specific order
        queue.push(market)
        queue.push(signal)
        queue.push(order)

        # Should pop in same order (FIFO)
        assert queue.pop() == market
        assert queue.pop() == signal
        assert queue.pop() == order

    def test_mixed_timestamps_and_fifo(self):
        """
        Mix of different timestamps with FIFO for same timestamp.
        """
        queue = EventQueue()
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)

        event1a = make_market_event(ts1, "SPY")
        event1b = make_signal_event(ts1, "SPY")
        event2a = make_market_event(ts2, "SPY")

        queue.push(event2a)  # Later timestamp first
        queue.push(event1a)  # Earlier timestamp
        queue.push(event1b)  # Same as event1a

        # Pop order: event1a, event1b (FIFO), event2a
        assert queue.pop() == event1a
        assert queue.pop() == event1b
        assert queue.pop() == event2a


class TestEventQueueErrors:
    """
    Test error conditions.
    """

    def test_pop_empty_raises(self):
        """
        Pop on empty queue should raise IndexError.
        """
        queue = EventQueue()
        with pytest.raises(IndexError, match="empty"):
            queue.pop()

    def test_peek_empty_raises(self):
        """
        Peek on empty queue should raise IndexError.
        """
        queue = EventQueue()
        with pytest.raises(IndexError, match="empty"):
            queue.peek()


class TestEventQueueMultipleOperations:
    """
    Test sequences of operations.
    """

    def test_push_pop_push_pop(self):
        """
        Interleaved push/pop operations.
        """
        queue = EventQueue()
        ts1 = datetime(2024, 1, 1)
        ts2 = datetime(2024, 1, 2)

        event1 = make_market_event(ts1)
        event2 = make_market_event(ts2)

        queue.push(event1)
        assert queue.pop() == event1

        queue.push(event2)
        assert queue.pop() == event2

        assert queue.is_empty()

    def test_many_events(self):
        """
        Queue should handle many events.
        """
        queue = EventQueue()
        base = datetime(2024, 1, 1)

        # Push 100 events in random order
        events = [make_market_event(base + timedelta(days=i)) for i in range(100)]
        for event in reversed(events):  # Push in reverse
            queue.push(event)

        # Should pop in timestamp order
        for expected in events:
            assert queue.pop() == expected
