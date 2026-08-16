"""Capture and verify AstrBot's model-bound prompt-cache structure.

Run from the repository root with:

    python scripts/verify_prompt_cache_structure.py
"""

import asyncio
import hashlib
import json
from typing import Any

from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.astr_main_agent import _append_dynamic_user_context
from astrbot.core.provider.entities import LLMResponse, ProviderRequest, TokenUsage
from astrbot.core.provider.provider import Provider


class CapturingMockModel(Provider):
    """Minimal model provider that captures the final AstrBot request boundary."""

    def __init__(self) -> None:
        super().__init__(
            {
                "id": "prompt-cache-structure-mock",
                "modalities": ["text", "tool_use"],
                "max_context_tokens": 0,
            },
            {},
        )
        self.payload: dict[str, Any] | None = None

    def get_current_key(self) -> str:
        return "mock"

    def set_key(self, key: str) -> None:
        del key

    async def get_models(self) -> list[str]:
        return ["prompt-cache-structure-mock"]

    async def text_chat(self, **kwargs: Any) -> LLMResponse:
        contexts = self._ensure_message_to_dicts(kwargs.get("contexts"))
        tools = kwargs.get("func_tool")
        self.payload = {
            "messages": contexts,
            "tools": tools.openai_schema() if tools else [],
        }
        return LLMResponse(
            role="assistant",
            completion_text="mock response",
            usage=TokenUsage(input_other=1, output=1),
        )


class NoOpToolExecutor:
    """Executor placeholder; the capturing model never requests a tool call."""

    @staticmethod
    def execute(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("The structure mock must not execute tools.")


class NoOpHooks(BaseAgentRunHooks):
    """Use the framework's no-op hook defaults."""


def _build_tools(reverse: bool) -> ToolSet:
    query_parameters = {
        "properties": {
            "query": {
                "description": "Search query",
                "type": "string",
            }
        },
        "required": ["query"],
        "type": "object",
    }
    read_parameters = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {
                "type": "string",
                "description": "File path",
            }
        },
    }
    tools = [
        FunctionTool(
            name="web_search",
            description="Search the web",
            parameters=query_parameters,
        ),
        FunctionTool(
            name="read_file",
            description="Read a file",
            parameters=read_parameters,
        ),
    ]
    return ToolSet(list(reversed(tools)) if reverse else tools)


async def capture_framework_payload(
    *,
    reverse_tools: bool,
    persona: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Run one request through the real runner and capture its provider payload."""
    request = ProviderRequest(
        prompt=user_prompt,
        system_prompt="Stable root system prompt.",
        contexts=[
            {"role": "user", "content": "Historical question"},
            {"role": "assistant", "content": "Historical answer"},
        ],
        func_tool=_build_tools(reverse_tools),
    )
    _append_dynamic_user_context(request, "persona", persona)
    _append_dynamic_user_context(request, "skills", "Available skill: repository audit")

    model = CapturingMockModel()
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=model,
        request=request,
        run_context=ContextWrapper(context=None),
        tool_executor=NoOpToolExecutor(),
        agent_hooks=NoOpHooks(),
        streaming=False,
    )
    async for _ in runner.step_until_done(1):
        pass

    if model.payload is None:
        raise AssertionError("The mock model did not receive a request.")
    return model.payload


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


async def verify_structure() -> dict[str, Any]:
    """Verify equivalent payloads and dynamic-user-context prefix placement."""
    baseline = await capture_framework_payload(
        reverse_tools=False,
        persona="Helpful persona version A.",
        user_prompt="User request A.",
    )
    reordered = await capture_framework_payload(
        reverse_tools=True,
        persona="Helpful persona version A.",
        user_prompt="User request A.",
    )
    dynamic_change = await capture_framework_payload(
        reverse_tools=True,
        persona="Helpful persona version B.",
        user_prompt="User request B.",
    )

    if _canonical_bytes(baseline) != _canonical_bytes(reordered):
        raise AssertionError("Equivalent requests produced unstable payloads.")

    baseline_prefix = {
        "messages": baseline["messages"][:-1],
        "tools": baseline["tools"],
    }
    changed_prefix = {
        "messages": dynamic_change["messages"][:-1],
        "tools": dynamic_change["tools"],
    }
    if _canonical_bytes(baseline_prefix) != _canonical_bytes(changed_prefix):
        raise AssertionError("Dynamic request data changed the reusable prefix.")

    current_user = baseline["messages"][-1]
    content = current_user.get("content")
    if current_user.get("role") != "user" or not isinstance(content, list):
        raise AssertionError("The current user message must use content blocks.")
    if len(content) < 3:
        raise AssertionError("Expected persona, skills, and raw user input blocks.")
    if 'name="persona"' not in content[0].get("text", ""):
        raise AssertionError("Persona context is not before the user input.")
    if 'name="skills"' not in content[1].get("text", ""):
        raise AssertionError("Skills context is not before the user input.")
    if content[2] != {"type": "text", "text": "User request A."}:
        raise AssertionError("Raw user input must follow all dynamic context blocks.")

    return {
        "equivalent_payload_sha256": _sha256(baseline),
        "stable_prefix_sha256": _sha256(baseline_prefix),
        "captured_payload": baseline,
    }


async def main() -> None:
    report = await verify_structure()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("PASS: AstrBot model-bound request structure is stable.")


if __name__ == "__main__":
    asyncio.run(main())
