from tianji_wuji_runtime.runtime.keyboard import RuntimeState, RuntimeStateMachine


def test_pause_aliases_and_reset_request() -> None:
    state_machine = RuntimeStateMachine(auto_start=True)

    state_machine.update("E")
    assert state_machine.state == RuntimeState.PAUSED
    state_machine.update("r")
    assert state_machine.state == RuntimeState.RUNNING
    state_machine.update("p")
    assert state_machine.state == RuntimeState.PAUSED

    state_machine.update("B")
    assert state_machine.consume_reset_request()
    assert not state_machine.consume_reset_request()
