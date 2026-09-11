from ..message import Message
from .persistent_context import is_context_message
from .round_utils import split_into_rounds


class ContextTruncator:
    """Context truncator."""

    def _has_tool_calls(self, message: Message) -> bool:
        """Check if a message contains tool calls."""
        return (
            message.role == "assistant"
            and message.tool_calls is not None
            and len(message.tool_calls) > 0
        )

    @staticmethod
    def _has_reply(round_messages: list[Message]) -> bool:
        """Check whether a round already holds the assistant's final reply.

        An assistant message that only calls tools is not a reply yet: a round
        still in its tool loop is the request being answered.
        """
        return any(
            message.role == "assistant" and not message.tool_calls
            for message in round_messages
        )

    @staticmethod
    def _split_system_rest(
        messages: list[Message],
    ) -> tuple[list[Message], list[Message]]:
        """Split messages into system messages and the rest.

        Returns:
            tuple: (system_messages, non_system_messages)
        """
        first_non_system = 0
        for i, msg in enumerate(messages):
            if msg.role != "system":
                first_non_system = i
                break
        return messages[:first_non_system], messages[first_non_system:]

    @staticmethod
    def _ensure_user_message(
        system_messages: list[Message],
        truncated: list[Message],
        original_messages: list[Message],
    ) -> list[Message]:
        """Ensure the result always contains the first user message right after
        system messages. This is required by many LLM APIs (e.g. Zhipu) that
        mandate a ``user`` message immediately following the ``system`` message.
        """
        if truncated and truncated[0].role == "user":
            return system_messages + truncated

        # Locate the first user message from the *original* list.
        first_user = next((m for m in original_messages if m.role == "user"), None)
        if first_user is None:
            return system_messages + truncated

        return system_messages + [first_user] + truncated

    def fix_messages(self, messages: list[Message]) -> list[Message]:
        """Fix the message list to ensure the validity of tool call and tool response pairing.

        This method ensures that:
        1. Each `tool` message is preceded by an `assistant` message containing `tool_calls`.
        2. Each `assistant` message containing `tool_calls` is followed by corresponding `

        This is a requirement of the OpenAI Chat Completions API specification (Gemini enforces this strictly).
        """
        if not messages:
            return messages

        fixed_messages: list[Message] = []
        pending_assistant: Message | None = None
        pending_tools: list[Message] = []

        def flush_pending_if_valid() -> None:
            nonlocal pending_assistant, pending_tools
            if pending_assistant is not None and pending_tools:
                fixed_messages.append(pending_assistant)
                fixed_messages.extend(pending_tools)
            pending_assistant = None
            pending_tools = []

        for msg in messages:
            if msg.role == "tool":
                # Only record tool responses when there is a pending assistant(tool_calls)
                if pending_assistant is not None:
                    pending_tools.append(msg)
                # Isolated tool messages without a preceding assistant(tool_calls) are ignored
                continue

            if self._has_tool_calls(msg):
                # When encountering a new assistant(tool_calls), first process the old pending chain
                flush_pending_if_valid()
                pending_assistant = msg
                continue

            # Non-tool messages that do not contain tool_calls will break the pending chain.
            # Flush any pending chain first, then append the current message normally.
            flush_pending_if_valid()
            fixed_messages.append(msg)

        # Flush the last pending chain at the end,
        # ensuring that any remaining valid assistant(tool_calls) and its tools are included in the final list.
        flush_pending_if_valid()

        return fixed_messages

    def truncate_by_turns(
        self,
        messages: list[Message],
        keep_most_recent_turns: int,
        drop_turns: int = 1,
    ) -> list[Message]:
        """
        Turn-based truncation strategy, which drops the oldest turns while keeping the most recent N turns.
        A turn starts at a user message, together with the context message right
        before it (anchors, message units), and runs through the assistant's
        reply and tool calls.
        This method ensures that the truncated context list conforms to OpenAI's context format.

        Args:
            messages: The original list of messages in the context.
            keep_most_recent_turns: The number of most recent turns to keep. If set to -1, it means keeping all turns (no truncation).
            drop_turns: The number of turns to drop from the beginning.

        Returns:
            The truncated list of messages.
        """
        if keep_most_recent_turns == -1:
            return messages

        system_messages, non_system_messages = self._split_system_rest(messages)
        rounds = split_into_rounds(non_system_messages)
        # The request still being answered is not a finished turn: it is neither
        # counted nor dropped.
        in_flight = rounds.pop() if rounds and not self._has_reply(rounds[-1]) else []

        if len(rounds) <= keep_most_recent_turns:
            return messages

        num_to_keep = keep_most_recent_turns - drop_turns + 1
        kept_rounds = rounds[-num_to_keep:] if num_to_keep > 0 else []
        truncated_contexts = [
            message for round_messages in kept_rounds for message in round_messages
        ] + in_flight

        # Find the first user message
        index = next(
            (i for i, item in enumerate(truncated_contexts) if item.role == "user"),
            None,
        )
        if index is not None and index > 0:
            truncated_contexts = truncated_contexts[index:]

        result = self._ensure_user_message(
            system_messages, truncated_contexts, messages
        )
        return self.fix_messages(result)

    def truncate_by_dropping_oldest_turns(
        self,
        messages: list[Message],
        drop_turns: int = 1,
    ) -> list[Message]:
        """Drop the oldest N turns, regardless of the number of turns to keep."""
        if drop_turns <= 0:
            return messages

        system_messages, non_system_messages = self._split_system_rest(messages)
        rounds = split_into_rounds(non_system_messages)
        # The request still being answered always survives.
        in_flight = rounds.pop() if rounds and not self._has_reply(rounds[-1]) else []
        truncated_non_system = [
            message
            for round_messages in rounds[drop_turns:]
            for message in round_messages
        ] + in_flight

        # Find the first user message
        index = next(
            (i for i, item in enumerate(truncated_non_system) if item.role == "user"),
            None,
        )
        if index is not None:
            truncated_non_system = truncated_non_system[index:]

        result = self._ensure_user_message(
            system_messages, truncated_non_system, messages
        )
        return self.fix_messages(result)

    def truncate_by_halving(
        self,
        messages: list[Message],
    ) -> list[Message]:
        """Halve the number of messages, keeping the most recent ones."""
        if len(messages) <= 2:
            return messages

        system_messages, non_system_messages = self._split_system_rest(messages)

        messages_to_delete = len(non_system_messages) // 2
        if messages_to_delete == 0:
            return messages

        # Start at the first user message after the cut, moving back over the
        # context messages right before it so a prompt keeps its context.
        start = next(
            (
                index
                for index in range(messages_to_delete, len(non_system_messages))
                if non_system_messages[index].role == "user"
            ),
            None,
        )
        if start is None:
            truncated_non_system = non_system_messages[messages_to_delete:]
        else:
            while start > 0 and is_context_message(non_system_messages[start - 1]):
                start -= 1
            truncated_non_system = non_system_messages[start:]

        result = self._ensure_user_message(
            system_messages, truncated_non_system, messages
        )
        return self.fix_messages(result)
