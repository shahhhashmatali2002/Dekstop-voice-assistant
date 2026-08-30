"""Doing the routing so the user does not have to - and always undoing it.

The user's words: "i will not change here any option ... when i start
application then update this all things from application and after close
should put all again on default so that i do not need to change again and
again."

The restore is the part that matters most. A machine left with its default
output pointing at a virtual cable has no sound anywhere, looks like broken
hardware, and the person who caused it is a program they already closed.
"""

import pytest

from app.audio.routing import DefaultOutputRouter, RoutingUnavailable


def build(previous: str = "speakers-id"):
    """A router wired to fakes, plus the log of what it set."""
    was_set: list[str] = []
    router = DefaultOutputRouter(
        setter=was_set.append,
        reader=lambda: previous,
        resolver=lambda name: f"{name}-id",
    )
    return router, was_set


def test_starting_sends_everything_to_the_cable():
    router, was_set = build()
    router.route("CABLE Input")
    assert was_set == ["CABLE Input-id"]
    assert router.routed is True


def test_stopping_puts_back_exactly_what_was_there():
    router, was_set = build(previous="headphones-id")
    router.route("CABLE Input")
    router.restore()
    assert was_set[-1] == "headphones-id"
    assert router.routed is False


def test_routing_twice_does_not_forget_the_original():
    """Otherwise the second route remembers the CABLE as the original, the
    restore puts the cable back, and the machine stays silent after Stop -
    which is the exact failure this whole module exists to avoid."""
    router, was_set = build(previous="headphones-id")
    router.route("CABLE Input")
    router.route("CABLE Input")
    router.restore()
    assert was_set[-1] == "headphones-id"


def test_restoring_twice_is_harmless():
    """It is called from stop, from shutdown, and from an atexit hook. All
    three can happen for one run."""
    router, was_set = build()
    router.route("CABLE Input")
    router.restore()
    before = list(was_set)
    router.restore()
    assert was_set == before


def test_restoring_without_routing_does_nothing():
    router, was_set = build()
    router.restore()
    assert was_set == []


def test_a_restore_never_raises():
    """It runs on the way out, including on the way out of a crash. An
    exception here would replace a real error with this one."""
    def explode(_):
        raise OSError("the device went away")

    router = DefaultOutputRouter(
        setter=explode, reader=lambda: "speakers-id", resolver=lambda n: "x"
    )
    with pytest.raises(OSError):
        router.route("CABLE Input")     # routing MAY fail loudly...
    router.restore()                     # ...restoring may not


def test_an_unknown_device_is_reported_not_guessed():
    def missing(_name):
        raise RoutingUnavailable("no playback device matching 'CABLE Input'")

    router = DefaultOutputRouter(
        setter=lambda _: None, reader=lambda: "speakers-id", resolver=missing
    )
    with pytest.raises(RoutingUnavailable):
        router.route("CABLE Input")
    assert router.routed is False, "a failed route must not claim to have routed"


# ----------------------------------------------------------------------
# reading the current default, whatever pycaw hands back
# ----------------------------------------------------------------------


def test_the_default_is_read_from_either_pycaw_shape(monkeypatch):
    """The bug, verbatim from the user's screen:

        Could not route the audio automatically (could not read the default
        output: 'AudioDevice' object has no attribute 'GetId')

    Some pycaw versions return the raw IMMDevice pointer, which has GetId();
    others return an AudioDevice wrapper, which has .id and no such method.
    """
    import app.audio.routing as routing

    class ComPointer:
        def GetId(self):
            return "{0.0.0.00000000}.{com}"

    class Wrapper:            # what the user's version returns
        id = "{0.0.0.00000000}.{wrapper}"

    for shape, expected in ((ComPointer(), "com"), (Wrapper(), "wrapper")):
        fake = type("AudioUtilities", (), {"GetSpeakers": staticmethod(lambda s=shape: s)})
        module = type("pycaw", (), {"AudioUtilities": fake})
        monkeypatch.setitem(
            __import__("sys").modules, "pycaw.pycaw", module
        )
        assert expected in routing.default_output_id()


def test_something_with_neither_says_so_clearly(monkeypatch):
    import app.audio.routing as routing

    class Neither:
        pass

    fake = type("AudioUtilities", (), {"GetSpeakers": staticmethod(lambda: Neither())})
    module = type("pycaw", (), {"AudioUtilities": fake})
    monkeypatch.setitem(__import__("sys").modules, "pycaw.pycaw", module)

    with pytest.raises(RoutingUnavailable) as caught:
        routing.default_output_id()
    assert "neither GetId() nor .id" in str(caught.value)
