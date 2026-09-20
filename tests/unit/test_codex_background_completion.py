"""后台命令跑完后，结果以发起人的身份作为一条消息回到会话。

固定播报的问题是 agent 根本不知道结果——不进历史、不能用自己的语气说、
也不能就结果继续动作。所以走真正的一轮：发起人的回合还在跑就插进去，
否则按原有规则排队。
"""

import pytest

from astrbot.core.agent.runners.codex import native, wake
from astrbot.core.agent.runners.codex.native import ACTIVE_TURNS, ActiveTurn

UMO = "aiocqhttp:GroupMessage:30003"


class _Ctx:
    @staticmethod
    def get_config(umo=None):
        return {"agent_runner": {"runner_type": "codex"}}

    @staticmethod
    async def send_message(session, chain):
        return True


@pytest.fixture(autouse=True)
def _clean():
    ACTIVE_TURNS.clear()
    yield
    ACTIVE_TURNS.clear()


@pytest.fixture
def _no_new_turn(monkeypatch):
    """记录是否退而起了新的一轮。"""
    calls: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        calls.append(event.get_sender_id())
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)
    return calls


@pytest.mark.asyncio
async def test_the_initiator_s_running_turn_absorbs_the_result(
    monkeypatch, _no_new_turn
):
    steered: list[tuple] = []

    async def _steer(umo, sender_id, turn_input, *, prompt=""):
        steered.append((umo, sender_id))
        return ""

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-1",
        exit_code=0,
        output="DONE",
    )

    assert steered == [(UMO, "20017")]
    assert _no_new_turn == [], "已经插进那一轮了，不该再起一轮"


@pytest.mark.asyncio
async def test_someone_else_s_turn_makes_it_queue(monkeypatch, _no_new_turn):
    """别人的回合在跑：插不进去，退回成新的一轮，由会话锁排队。"""

    async def _steer(umo, sender_id, turn_input, *, prompt=""):
        return None  # try_steer 对不同发送者就是这样返回的

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-2",
        exit_code=0,
        output="DONE",
    )

    # 新的一轮仍然以发起人的身份运行，权限规则才对得上
    assert _no_new_turn == ["20017"]


@pytest.mark.asyncio
async def test_the_result_reaches_the_model(monkeypatch):
    seen: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        seen.append(prompt)
        return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    async def _steer(*a, **k):
        return None

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-3",
        exit_code=7,
        output="BACKGROUND_DONE_7741",
    )

    prompt = seen[0]
    assert "BACKGROUND_DONE_7741" in prompt
    assert 'exit_code="7"' in prompt
    assert 'session="sess-3"' in prompt
    # 别让它以为该重跑一遍
    assert "Do not start the command again" in prompt


@pytest.mark.asyncio
async def test_a_silent_turn_still_reports_the_exit_code(monkeypatch):
    """那一轮没产出回复时，至少要让用户知道命令结束了。"""
    sent: list[str] = []

    async def _run(ctx, event, cfg, prompt, delivery):
        return False

    class _Recording(_Ctx):
        @staticmethod
        async def send_message(session, chain):
            sent.append(chain.get_plain_text())
            return True

    monkeypatch.setattr(wake, "run_in_session_thread", _run)

    async def _steer(*a, **k):
        return None

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Recording(),
        session_str=UMO,
        sender_id="20017",
        role="member",
        session_id="sess-4",
        exit_code=1,
        output="",
    )

    assert sent and "sess-4" in sent[0] and "1" in sent[0]


@pytest.mark.asyncio
async def test_an_unknown_sender_skips_straight_to_a_new_turn(
    monkeypatch, _no_new_turn
):
    """没有发起人身份时不该去插队——那会插进别人的回合。"""
    ACTIVE_TURNS[UMO] = ActiveTurn(object(), "thread-1", "turn-1", "20019")
    ACTIVE_TURNS[UMO].ready.set()
    steered = []

    async def _steer(*a, **k):
        steered.append(a)
        return ""

    monkeypatch.setattr(native, "try_steer", _steer)

    await wake.run_background_exec_completion(
        _Ctx(),
        session_str=UMO,
        sender_id="",
        role="member",
        session_id="sess-5",
        exit_code=0,
        output="",
    )

    assert steered == [], "没有身份就去插队，等于插进别人的回合"
    assert len(_no_new_turn) == 1, "仍然要起一轮把结果说出来"
