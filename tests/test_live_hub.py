"""LiveHub bridge tests: no network. Events are injected directly as the feed thread would."""
from volscape.live.feed import Status
from volscape.live.hub import LiveHub
from volscape.live.state import parse_message
from tests.test_live_state import depth_msg, trade_msg


def test_hub_is_inert_until_started():
    hub = LiveHub()
    snap = hub.snapshot()
    assert not snap.connected and snap.status == "OFF"
    assert snap.last is None and snap.mid is None and snap.bids == [] and snap.quotes == []


def test_hub_applies_current_generation_and_ignores_stale():
    hub = LiveHub()
    hub.gen = 2
    hub._on_event(2, parse_message(depth_msg(bids=(("100", "3"), ("99", "1")), asks=(("101", "2"),))))
    hub._on_event(2, parse_message(trade_msg(tid=1, p="100.5", q="2")))
    hub._on_event(1, parse_message(trade_msg(tid=2, p="999", q="5")))       # stale generation
    hub._on_status(1, Status.LIVE, "old")                                     # stale status
    snap = hub.snapshot()
    assert snap.last == 100.5 and snap.mid == 100.5 and snap.spread == 1.0
    assert snap.bids[0] == (100.0, 3.0) and snap.asks == [(101.0, 2.0)]
    assert snap.imbalance == (4 - 2) / 6
    assert len(snap.trades) == 1 and snap.trades[0][3] == "BUY"
    assert snap.status == "OFF"


def test_reconnect_drops_book_and_event_log_dedupes():
    hub = LiveHub()
    hub.gen = 1
    hub._on_status(1, Status.LIVE, "wss://x")
    hub._on_event(1, parse_message(depth_msg()))
    for _ in range(5):
        hub._on_status(1, Status.RECONNECTING, "retry in 1.0s")
    snap = hub.snapshot()
    assert snap.bids == [] and snap.gaps == 1          # stale book dropped, gap counted once
    assert sum("RECONNECTING" in m for _, m in snap.events) == 1
