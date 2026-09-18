"""AstrBot skills for the Codex runner.

Codex threads run without a local execution environment by default, so the
model cannot `cat` a SKILL.md. Skills are listed compactly in the prompt and
read on demand through the ``astrbot_read_skill`` tool, which only opens
files inside the directory of a skill offered to the current event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.skills.skill_manager import SkillInfo

READ_SKILL_TOOL = "astrbot_read_skill"
MAX_READ_BYTES = 64 * 1024


def build_codex_skills_prompt(skills: list[SkillInfo]) -> str:
    lines = [
        f"- {skill.name}: {(skill.description or 'Read SKILL.md for details.').strip()}"
        for skill in skills
    ]
    return (
        "## Skills\n"
        "Skills are instruction bundles. Use one when the user names it or the task "
        "clearly matches its description:\n"
        + "\n".join(lines)
        + "\n\nBefore using a skill, read it with "
        f"`tools.astrbot__{READ_SKILL_TOOL}({{name}})`; pass `path` to open a file it "
        "references (e.g. `scripts/run.py`). Do not guess a skill's content."
    )


def _skill_dir(skill: SkillInfo) -> Path | None:
    path = Path(skill.path)
    if not skill.local_exists or not path.exists():
        return None
    return path.parent if path.is_file() else path


def read_skill_file(skills: list[SkillInfo], name: str, rel_path: str = "") -> str:
    skill = next((s for s in skills if s.name == name), None)
    if skill is None:
        available = ", ".join(s.name for s in skills) or "none"
        return f"error: unknown skill {name!r}. Available: {available}"
    base = _skill_dir(skill)
    if base is None:
        return f"error: skill {name!r} is not available on this host."
    rel = PureWindowsPath(rel_path or "SKILL.md")
    # Reject before touching the filesystem: absolute / drive / UNC paths
    # would otherwise be opened by resolve() (e.g. an outbound SMB request).
    if rel.is_absolute() or rel.drive or rel.anchor or ".." in rel.parts:
        return "error: path must be relative to the skill directory."
    target = (base / Path(*rel.parts)).resolve()
    base = base.resolve()
    if target != base and base not in target.parents:
        return "error: path escapes the skill directory."
    if target.is_dir():
        entries = sorted(
            f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()
        )
        return f"Directory {target.relative_to(base) or '.'}:\n" + "\n".join(entries)
    if not target.exists():
        return f"error: {rel_path or 'SKILL.md'} not found in skill {name!r}."
    with target.open("rb") as f:
        data = f.read(MAX_READ_BYTES)
    text = data.decode("utf-8", errors="replace")
    if target.stat().st_size > MAX_READ_BYTES:
        text += f"\n...[truncated at {MAX_READ_BYTES} bytes]"
    return text


@dataclass
class ReadSkillTool(FunctionTool):
    name: str = READ_SKILL_TOOL
    description: str = (
        "Read a skill's SKILL.md, or a file / directory inside that skill "
        "(relative `path`, e.g. `scripts/run.py`)."
    )
    parameters: dict = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name."},
                "path": {
                    "type": "string",
                    "description": "File or directory inside the skill; default SKILL.md.",
                },
            },
            "required": ["name"],
        }
    )

    async def call(self, context: ContextWrapper, **kwargs: Any) -> str:
        event = context.context.event  # type: ignore[attr-defined]
        skills = event.get_extra("_codex_skills") or []
        return read_skill_file(
            skills, str(kwargs.get("name") or ""), str(kwargs.get("path") or "")
        )
