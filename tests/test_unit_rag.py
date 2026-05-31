from src.app.rag import format_context


def test_format_context_sources_and_text():
    retrieved = [
        {"text": "Hello", "source": "a.md"},
        {"text": "World", "source": "b.md"},
        {"text": "Again", "source": "a.md"},
    ]
    context, sources = format_context(retrieved)
    assert "Hello" in context
    assert "World" in context
    assert sources == ["a.md", "b.md"]