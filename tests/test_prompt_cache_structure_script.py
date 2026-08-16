import pytest

from scripts.verify_prompt_cache_structure import verify_structure


@pytest.mark.asyncio
async def test_mock_model_verifies_framework_prompt_cache_structure():
    report = await verify_structure()

    assert len(report["equivalent_payload_sha256"]) == 64
    assert len(report["stable_prefix_sha256"]) == 64
    assert report["captured_payload"]["messages"][0] == {
        "role": "system",
        "content": "Stable root system prompt.",
    }
