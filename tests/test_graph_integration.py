from langchain_core.runnables import RunnableLambda

from src.app.graph import build_graph_app, VerifyResult


class DummyDoc:
    def __init__(self, text: str, source: str):
        self.page_content = text
        self.metadata = {"source": source}


def test_graph_runs_end_to_end():
    # Fake retriever: always returns one doc
    def fake_retrieve(_q: str):
        return [DummyDoc("LCEL is prompt | model | parser.", "notes.md")]

    retriever = RunnableLambda(fake_retrieve)

    class FakeModel:
        # Used by the answer chain: prompt | model | StrOutputParser()
        def __call__(self, _inp):
            return "LCEL is a pipeline: prompt | model | parser."

        # Used by the verify chain: model.with_structured_output(VerifyResult)
        def with_structured_output(self, _schema):
            def _verify(_inp):
                return VerifyResult(confidence=0.99, decision="ok", rationale="supported")

            return RunnableLambda(_verify)

    run_graph = build_graph_app(
        retriever=retriever,
        model=FakeModel(),  # type: ignore
        verify_threshold=0.7,
        max_iterations=3,
    )

    out = run_graph("What is LCEL?")
    assert "LCEL" in out["answer"]
    assert out["decision"] == "ok"