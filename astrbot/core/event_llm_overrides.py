from __future__ import annotations

from typing import Any

SELECTED_PERSONA_EXTRA_KEY = "selected_persona"
SELECTED_PROVIDER_EXTRA_KEY = "selected_provider"
SELECTED_MODEL_EXTRA_KEY = "selected_model"


def normalize_event_override_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def get_event_selected_persona_id(event: Any) -> str | None:
    try:
        if event is None or not hasattr(event, "get_extra"):
            return None
        return normalize_event_override_id(event.get_extra(SELECTED_PERSONA_EXTRA_KEY))
    except Exception:
        return None
