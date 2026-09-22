import base64

from astrbot.core.platform.sources.mumble.text import (
    InlineImage,
    ParsedText,
    escape_text,
    html_to_text,
    image_html,
    markdown_to_html,
)

PNG = b"\x89PNG\r\n\x1a\nfake"


def test_official_client_plain_text():
    # MainWindow::sendChatbarText with Markdown off.
    sent = "a&nbsp;&lt;b&gt;&nbsp;&amp;&nbsp;c<br>line&nbsp;&nbsp;two"
    assert html_to_text(sent) == ParsedText("a <b> & c\nline  two")


def test_rich_html_and_links():
    message = (
        "<p>Hello <b>world</b></p><ul><li>one</li><li>two</li></ul>"
        '<p><a href="https://example.com">https://example.com</a> and '
        '<a href="https://docs.example.com/x">docs</a></p>'
        "<style>p{}</style>"
    )
    assert html_to_text(message).text == (
        "Hello world\n- one\n- two\nhttps://example.com and docs (https://docs.example.com/x)"
    )


def test_plain_text_passes_through():
    assert html_to_text("just text, 1 < 2").text == "just text, 1 < 2"


def test_inline_images_are_extracted():
    encoded = base64.b64encode(PNG).decode()
    message = f'look<br><img src="data:image/png;base64,{encoded}"/><img src="https://x/y.png">'
    assert html_to_text(message) == ParsedText(
        "look\n[image: https://x/y.png]", [InlineImage("image/png", PNG)]
    )
    assert html_to_text(image_html(PNG)).images == [InlineImage("image/png", PNG)]


def test_escape_text_roundtrips():
    text = "if a < b && c:\n    return  1"
    html = escape_text(text)
    assert html == "if a &lt; b &amp;&amp; c:<br> &nbsp;&nbsp;&nbsp;return &nbsp;1"
    assert html_to_text(html).text == text


def test_markdown_subset():
    reply = (
        "# Title\n**bold** and *it* and `a<b` and ~~no~~\n"
        "see [docs](https://example.com/d) or https://example.com\n"
        "```python\nprint('<x>')\n```\ndone"
    )
    assert markdown_to_html(reply) == (
        "<b>Title</b><br><b>bold</b> and <i>it</i> and <code>a&lt;b</code> and <s>no</s><br>"
        'see <a href="https://example.com/d">docs</a> or '
        '<a href="https://example.com">https://example.com</a><br>'
        "<pre>print('&lt;x&gt;')</pre>done"
    )
    assert markdown_to_html("`**not bold**`") == "<code>**not bold**</code>"
    assert markdown_to_html("a*b*c 2*3*4") == "a*b*c 2*3*4"
