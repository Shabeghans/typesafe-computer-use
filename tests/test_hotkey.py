from typesafe_computer_use.hotkey import HoldKey, key_state

RIGHT_OPTION, LEFT_OPTION = 61, 58


def test_only_the_chosen_key_counts():
    assert key_state("right_option", RIGHT_OPTION, 0x80040) is True
    assert key_state("right_option", RIGHT_OPTION, 0x80000) is False
    assert key_state("right_option", LEFT_OPTION, 0x80020) is None


def test_down_and_up_fire_once_each():
    events = []
    key = HoldKey("right_option", on_down=lambda: events.append("down"), on_up=lambda: events.append("up"))
    key.handle(RIGHT_OPTION, 0x80040)
    key.handle(LEFT_OPTION, 0x80060)  # left option joins: still held
    key.handle(RIGHT_OPTION, 0x80040)  # a repeat of the same state
    key.handle(RIGHT_OPTION, 0x0)
    assert events == ["down", "up"]
