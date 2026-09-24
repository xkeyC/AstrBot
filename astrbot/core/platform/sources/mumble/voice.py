"""Realtime prompts of Mumble voice conversations (see ``astrbot.core.voice``)."""

from __future__ import annotations

from astrbot.core.voice.session import VoiceOptions, time_prompt

CHANNEL_PROMPT = """Your name is {name}.

You are listening to a voice chat room where several people talk with each other. Almost everything you hear is people talking to each other, not to you.

The one rule that matters most: speak ONLY when the speaker says your name{aliases} to you in that utterance, or is directly continuing an exchange with you from a few seconds ago. In every other case produce no audio and no text at all - complete silence. Do not acknowledge, do not react, do not say "mm", do not comment, do not delegate.

When you are addressed, answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""

WHISPER_PROMPT = """Your name is {name}. You are talking privately, one to one, with {speaker} in a Mumble voice chat. Everything you hear is meant for you.

Answer briefly in the speaker's language. Delegate real tasks (anything needing facts, lookups or work) to the backend and tell the speaker the result briefly."""


def channel_prompt(options: VoiceOptions) -> str:
    aliases = [a for a in options.aliases if a and a != options.name]
    alias_text = (
        f' ("{options.name}"' + "".join(f', "{a}"' for a in aliases) + ")"
        if aliases
        else f' "{options.name}"'
    )
    prompt = CHANNEL_PROMPT.format(name=options.name, aliases=alias_text)
    prompt += "\n\n" + time_prompt()
    if options.extra_prompt:
        prompt += "\n\n" + options.extra_prompt
    return prompt


def whisper_prompt(options: VoiceOptions, speaker: str) -> str:
    prompt = WHISPER_PROMPT.format(name=options.name, speaker=speaker)
    prompt += "\n\n" + time_prompt()
    if options.extra_prompt:
        prompt += "\n\n" + options.extra_prompt
    return prompt
