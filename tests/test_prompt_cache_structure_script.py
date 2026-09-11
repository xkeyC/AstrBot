import pytest

from scripts.verify_prompt_cache_structure import verify_structure


@pytest.mark.asyncio
async def test_multi_turn_conversation_keeps_a_reusable_prompt_prefix():
    reports = await verify_structure()

    assert [report["turn"] for report in reports] == [1, 2, 3, 4, 5, 6]
    # Every request after the first reuses most of what the previous one sent.
    for report in reports[1:]:
        if not report["truncated"]:
            assert report["reused_messages"] == report["expected_reuse"]
            assert report["reuse_ratio"] > 0.5
