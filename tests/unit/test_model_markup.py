import random

import pytest

from astrbot.core.utils.model_markup import ModelMarkupStream, strip_model_markup

OPEN, CLOSE, SEP = chr(0xE200), chr(0xE201), chr(0xE202)
REF = f"{OPEN}cite{SEP}turn0search6{SEP}turn1search4{SEP}turn2search0{CLOSE}"

CASES = [
    # ChatGPT content references, with and without private-use framing.
    (
        f"原生整合是它最大的优势。{REF} 5. 如果只是聊天",
        "原生整合是它最大的优势。 5. 如果只是聊天",
    ),
    (
        "最大的优势。citeturn0search6turn1search4turn2search0 5. 没有必要。citeturn2search0turn2search1",
        "最大的优势。 5. 没有必要。",
    ),
    (f"见{OPEN}filecite{SEP}turn0file0{CLOSE}。", "见。"),
    (f'{OPEN}entity{SEP}["company","Google"]{CLOSE} 很强', "Google 很强"),
    (f"{OPEN}entity{SEP}坏的{CLOSE}!", "!"),
    (f"尾部没闭合 {OPEN}cite{SEP}turn0search1", "尾部没闭合"),
    ("旧式引用【F:README.md†L5-L14】结束", "旧式引用结束"),
    ("普通括号【注意】保留", "普通括号【注意】保留"),
    # Memory citations.
    (
        "答案。\n\n<oai-mem-citation>\n<citation_entries>\nMEMORY.md:1-2|note=[x]\n"
        "</citation_entries>\n</oai-mem-citation>",
        "答案。",
    ),
    ("答案 <oai-mem-citation>未闭合", "答案"),
    # Codex App directives.
    (
        '报表已生成 :codex-file-citation{path="/tmp/report.xlsx" purpose="output"}。',
        "报表已生成 。",
    ),
    (
        'Done\n\n::git-stage{cwd="/repo"} ::git-push{cwd="/repo" branch="feat/x"}',
        "Done",
    ),
    (
        '::git-create-pr{cwd="C:\\repo" branch="feature/{rollout}" isDraft=true}',
        "",
    ),
    (
        '::code-comment{title="空指针" body="这里要判空。" file="src/a.py" start=3 end=5 priority=1}',
        "- [P1] 空指针 — src/a.py:3-5\n  这里要判空。",
    ),
    # Things that must stay.
    (
        "时间 10:30，网址 https://example.com/a:b{c}",
        "时间 10:30，网址 https://example.com/a:b{c}",
    ),
    ("表情 :smile: 和 a::b", "表情 :smile: 和 a::b"),
    ("I cite the turn of events; recite it.", "I cite the turn of events; recite it."),
    (
        '```\n::git-push{cwd="/repo"}\ncitation【a†b】\n```\n外面 ::git-push{cwd="/r"}',
        '```\n::git-push{cwd="/repo"}\ncitation【a†b】\n```\n外面',
    ),
    ("  缩进保留\n- 列表", "  缩进保留\n- 列表"),
]


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_strip_model_markup(source, expected):
    assert strip_model_markup(source) == expected


def _stream(chunks):
    stream = ModelMarkupStream()
    return "".join(stream.push(c) for c in chunks) + stream.finish()


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_stream_matches_one_shot_for_any_split(source, expected):
    whole = _stream([source])
    for cut in range(len(source) + 1):
        assert _stream([source[:cut], source[cut:]]) == whole, cut
    rng = random.Random(len(source))
    for _ in range(20):
        cuts = sorted(rng.sample(range(len(source) + 1), min(4, len(source))))
        parts = [source[a:b] for a, b in zip([0, *cuts], [*cuts, len(source)])]
        assert _stream(parts) == whole, parts


def test_stream_emits_plain_text_without_waiting_for_line_end():
    stream = ModelMarkupStream()
    assert stream.push("你好，世界") == "你好，世界"
    assert stream.push("。结论") == "。结论"
    # A possible citation start is held until it resolves.
    assert stream.push(" cite") == " "
    assert stream.push("turn0search1") == ""
    assert stream.push(" 下一句") == "下一句"
    assert stream.finish() == ""
