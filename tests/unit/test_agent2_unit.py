from __future__ import annotations

import json

import pytest
from pydantic import BaseModel


def test_search_in_related_disciplines_requires_init():
    import src.agents.agent2 as agent2

    agent2.PARAMS_2.qdrant_service = None
    agent2.PARAMS_2.collection_name = None

    with pytest.raises(RuntimeError, match="Qdrant не инициализирован"):
        agent2._rag_search_and_rerank(["q"], ["q"], output_model=BaseModel)


def test_rag_search_node_uses_rag_and_reranker_prompts(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    captured: dict[str, object] = {}

    def fake_search(rp: list[str], rr: list[str], _m):
        captured["rag_prompts"] = rp
        captured["reranker_prompts"] = rr
        return ["c1", "c2"]

    monkeypatch.setattr(agent2, "_rag_search_and_rerank", fake_search)
    monkeypatch.setattr(agent2, "format_rag_context", lambda chunks: "CTX:" + "|".join(chunks))

    out = agent2.rag_search_node(
        {
            "input_query": "hello",
            "rag_prompts": ["dense query", "dense query 2", "dense query 3"],
            "reranker_prompts": ["rerank query", "rerank query 2", "rerank query 3"],
            "rag_context": "",
            "rag_contexts": [],
            "output_model": BaseModel,
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    
    assert captured == {
        "rag_prompts": ["dense query", "dense query 2", "dense query 3"],
        "reranker_prompts": ["rerank query", "rerank query 2", "rerank query 3"],
    }
    assert out["rag_context"] == "CTX:c1|c2"


def test_generate_retrieval_prompts_node(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int | None = None

    class FakePrompts:
        rag_prompts = [
            agent2.RagPrompt(rag_prompt="  dense1  "),
            agent2.RagPrompt(rag_prompt="  dense2  "),
            agent2.RagPrompt(rag_prompt="  dense3  "),
        ]
        reranker_prompts = [
            agent2.RerankPrompt(reranker_prompt="  rerank1  "),
            agent2.RerankPrompt(reranker_prompt="  rerank2  "),
            agent2.RerankPrompt(reranker_prompt="  rerank3  "),
        ]

    class FakeStructured:
        def invoke(self, _messages):
            return FakePrompts()

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    agent2.PARAMS_2.llm = FakeLLM()

    out = agent2.generate_retrieval_prompts_node(
        {
            "input_query": "user q\nextract x",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "",
            "rag_contexts": [],
            "output_model": Out,
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    assert out["rag_prompts"] == ["dense1", "dense2", "dense3"]
    assert out["reranker_prompts"] == ["rerank1", "rerank2", "rerank3"]


def test_generate_retrieval_prompts_increments_count_after_rewrite(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int | None = None

    class FakePrompts:
        rag_prompts = [
            agent2.RagPrompt(rag_prompt="a"),
            agent2.RagPrompt(rag_prompt="b"),
            agent2.RagPrompt(rag_prompt="c"),
        ]
        reranker_prompts = [
            agent2.RerankPrompt(reranker_prompt="r1"),
            agent2.RerankPrompt(reranker_prompt="r2"),
            agent2.RerankPrompt(reranker_prompt="r3"),
        ]

    class FakeStructured:
        def invoke(self, _messages):
            return FakePrompts()

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    agent2.PARAMS_2.llm = FakeLLM()

    out = agent2.generate_retrieval_prompts_node(
        {
            "input_query": "q",
            "rag_prompts": ["old_r"],
            "reranker_prompts": ["old_rr"],
            "rag_context": "",
            "rag_contexts": [],
            "output_model": Out,
            "answer": "{}",
            "check_decision": "REWRITE",
            "check_reason": "мало данных",
            "rewrite_focus": "нормы",
            "rewrite_count": 0,
        }
    )
    assert out["rewrite_count"] == 1
    assert out["rag_prompts"] == ["a", "b", "c"]


def test_rag_search_passes_part_names_to_qdrant(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int | None = None

    captured: dict[str, object] = {}

    class FakeQdrant:
        def run_query(self, query: str, collection_name: str, limit: int = 3, part_names=None):
            captured["query"] = query
            captured["collection_name"] = collection_name
            captured["limit"] = limit
            captured["part_names"] = part_names

            class P:
                id = "p1"
                score = 0.9
                payload = {"text": "t1"}

            return [P()]

    agent2.PARAMS_2.qdrant_service = FakeQdrant()
    agent2.PARAMS_2.collection_name = "main"

    monkeypatch.setattr(
        agent2,
        "rerank_with_expanded_queries",
        lambda _qs, texts, **kwargs: [(texts[0], 1.0)],
    )
    monkeypatch.setattr(agent2, "get_part_names_for_model", lambda _m: ["АР", "КР"])

    chunks = agent2._rag_search_and_rerank(
        ["q1", "q2", "q3"], ["rr1", "rr2", "rr3"], output_model=Out
    )
    assert chunks == ["t1"]
    assert captured["part_names"] == ["АР", "КР"]
    assert captured["collection_name"] == "main"
    assert captured["limit"] == 50


def test_answer_node_happy_path(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int

    class FakeStructured:
        def invoke(self, _messages):
            return Out(x=1)

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    agent2.PARAMS_2.llm = FakeLLM()

    out = agent2.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "ctx",
            "rag_contexts": ["ctx1", "ctx2"],
            "output_model": Out,
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    assert out["answer"] == Out(x=1).model_dump_json()


def test_answer_node_fallback1_uses_validate_and_dump(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int | None = None

    class FakeStructured:
        def invoke(self, _messages):
            raise ValueError("boom")

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

        def invoke(self, _messages):
            class R:
                content = '{"x": 2}'

            return R()

    agent2.PARAMS_2.llm = FakeLLM()
    monkeypatch.setattr(agent2, "validate_and_dump_json_str", lambda _m, _s: '{"x": 2}')

    out = agent2.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "ctx",
            "rag_contexts": ["ctx1", "ctx2"],
            "output_model": Out,
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    assert out["answer"] == '{"x": 2}'


def test_answer_node_fallback2_returns_empty_dict_json(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: int | None = None

    class FakeStructured:
        def invoke(self, _messages):
            raise ValueError("boom")

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

        def invoke(self, _messages):
            raise RuntimeError("also boom")

    agent2.PARAMS_2.llm = FakeLLM()

    out = agent2.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "ctx",
            "rag_contexts": ["ctx1", "ctx2"],
            "output_model": Out,
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    assert out["answer"] == "{}"


def test_answer_node_postprocess_fills_from_merged(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class Out(BaseModel):
        x: list
        y: str

    class FakeStructured:
        def invoke(self, _messages):
            return Out(x=['5'], y="")

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    agent2.PARAMS_2.llm = FakeLLM()

    out = agent2.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "ctx",
            "rag_contexts": ["ctx1", "ctx2"],
            "output_model": Out,
            "answer": '{"x":[],"y":"keep"}',
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "rewrite_count": 0,
        }
    )
    data = json.loads(out["answer"])
    assert data["x"] == ['5']
    assert data["y"] == "keep"


def test_route_after_check_limits_rewrites():
    import src.agents.agent2 as agent2

    assert agent2.route_after_check({"check_decision": "REWRITE", "rewrite_count": 0}) == "generate_retrieval_prompts_node"
    assert agent2.route_after_check({"check_decision": "REWRITE", "rewrite_count": 1}) == "generate_retrieval_prompts_node"
    assert agent2.route_after_check({"check_decision": "REWRITE", "rewrite_count": 2}) == agent2.END
    assert agent2.route_after_check({"check_decision": "OK", "rewrite_count": 0}) == agent2.END


def test_init_graph_2_raises_when_collection_unknown_and_no_project_parts_path(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    class FakeClient:
        def collection_exists(self, _name: str) -> bool:
            return False

    class FakeQdrant:
        def __init__(self):
            self.client = FakeClient()

    monkeypatch.setattr(agent2, "build_qdrant_service", lambda: FakeQdrant())

    with pytest.raises(ValueError, match="project_parts_path не передан"):
        agent2.init_graph_2(collection_name="does-not-exist", project_parts_path=None)


def test_init_graph_2_existing_collection_does_not_create(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent2 as agent2

    called = {"create": 0, "fill": 0, "parts": 0}

    class FakeClient:
        def collection_exists(self, _name: str) -> bool:
            return True

    class FakeQdrant:
        def __init__(self):
            self.client = FakeClient()

    class FakeLlmModel:
        def __init__(self, *args, **kwargs):
            pass

        def create(self):
            return object()

    class FakeStateGraph:
        def __init__(self, _state_type):
            pass

        def add_node(self, *_args, **_kwargs):
            return None

        def add_edge(self, *_args, **_kwargs):
            return None

        def add_conditional_edges(self, *_args, **_kwargs):
            return None

        def compile(self):
            return "compiled"

    monkeypatch.setattr(agent2, "build_qdrant_service", lambda: FakeQdrant())
    monkeypatch.setattr(agent2, "create_project_parts", lambda _p: called.__setitem__("parts", called["parts"] + 1))
    monkeypatch.setattr(agent2, "create_collection", lambda *_a, **_k: called.__setitem__("create", called["create"] + 1))
    monkeypatch.setattr(agent2, "fill_collection", lambda *_a, **_k: called.__setitem__("fill", called["fill"] + 1))
    monkeypatch.setattr(agent2, "LlmModel", FakeLlmModel)
    monkeypatch.setattr(agent2, "StateGraph", FakeStateGraph)

    compiled = agent2.init_graph_2(collection_name="main", project_parts_path=None)
    assert compiled == "compiled"
    assert called == {"create": 0, "fill": 0, "parts": 0}
    assert agent2.PARAMS_2.collection_name == "main"
    assert agent2.PARAMS_2.qdrant_service is not None
    assert agent2.PARAMS_2.llm is not None

