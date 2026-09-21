"""A key held down anywhere on the system: voice mode's push to talk.

A Quartz event tap sees each modifier key change, whichever app is in front. The keys offered are
modifiers held alone, so holding one types nothing. The tap needs the terminal to have
Accessibility permission, which driving the machine needs anyway.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import Quartz

# name: (key code, the device-dependent flag that is set while that one key is down)
KEYS = {
    "right_option": (61, 0x40),
    "right_command": (54, 0x10),
    "fn": (63, Quartz.kCGEventFlagMaskSecondaryFn),
}
DEFAULT_KEY = "right_option"


def key_state(key: str, keycode: int, flags: int) -> bool | None:
    """Whether a modifier change leaves `key` down, or None when the change is to another key."""
    code, mask = KEYS[key]
    if keycode != code:
        return None
    return bool(flags & mask)


class HoldKey:
    """Calls on_down when `key` goes down and on_up when it comes up, from the tap's own thread."""

    def __init__(self, key: str, on_down: Callable[[], None], on_up: Callable[[], None]):
        self.key = key
        self.on_down = on_down
        self.on_up = on_up
        self.down = False
        self._tap = None
        self._loop = None

    def handle(self, keycode: int, flags: int) -> None:
        state = key_state(self.key, keycode, flags)
        if state is None or state == self.down:
            return
        self.down = state
        (self.on_down if state else self.on_up)()

    def start(self) -> None:
        """Start watching. Raises PermissionError when macOS refuses the tap."""
        ready = threading.Event()
        threading.Thread(target=self._watch, args=(ready,), name="hotkey", daemon=True).start()
        ready.wait(5)
        if self._tap is None:
            raise PermissionError("cannot watch the keyboard: grant this terminal Accessibility in System Settings")

    def _watch(self, ready: threading.Event) -> None:
        def callback(proxy, kind, event, refcon):
            if kind in (Quartz.kCGEventTapDisabledByTimeout, Quartz.kCGEventTapDisabledByUserInput):
                Quartz.CGEventTapEnable(self._tap, True)  # macOS pauses a slow tap; carry on
            elif kind == Quartz.kCGEventFlagsChanged:
                self.handle(
                    Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode), Quartz.CGEventGetFlags(event)
                )
            return event

        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
            callback,
            None,
        )
        if self._tap is None:
            ready.set()
            return
        self._loop = Quartz.CFRunLoopGetCurrent()
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        ready.set()
        Quartz.CFRunLoopRun()

    def stop(self) -> None:
        if self._loop is not None:
            Quartz.CFRunLoopStop(self._loop)
