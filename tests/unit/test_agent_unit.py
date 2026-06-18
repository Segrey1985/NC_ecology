from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from config.config_file import build_runtime_config


def _resources(agent, *, llm=None, qdrant_service=None, collection_name="main", runtime_cfg=None):
    return agent.GraphResources(
        collection_name=collection_name,
        qdrant_service=qdrant_service,
        llm=llm,
        runtime_cfg=runtime_cfg or build_runtime_config("on"),
    )


def _config(output_model: type[BaseModel]):
    return {
        "configurable": {"output_model": output_model},
        "metadata": {"chapter_module_path": "tests.fake_chapter"},
    }


def test_search_in_related_disciplines_requires_init():
    import src.agents.agent as agent

    with pytest.raises(RuntimeError, match="Qdrant не инициализирован"):
        agent._rag_search_and_rerank(
            _resources(agent, qdrant_service=None, collection_name=None),
            ["q"],
            ["q"],
            output_model=BaseModel,
            chapter_module_path="tests.fake_chapter",
        )


def test_rag_search_node_uses_rag_and_reranker_prompts(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

    captured: dict[str, object] = {}

    def fake_search(_resources, rp: list[str], rr: list[str], _m, _path: str):
        captured["chunks_all"] = rp
        captured["reranker_prompts"] = rr
        return ["c1", "c2"]

    monkeypatch.setattr(agent, "_rag_search_and_rerank", fake_search)

    out = agent.rag_search_node(
        {
            "input_query": "hello",
            "rag_prompts": ["dense query", "dense query 2", "dense query 3"],
            "reranker_prompts": ["rerank query", "rerank query 2", "rerank query 3"],
            "chunks": [""],
            "chunks_all": [],
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _config(BaseModel),
        _resources(agent),
    )
    
    assert captured == {
        "chunks_all": ["dense query", "dense query 2", "dense query 3"],
        "reranker_prompts": ["rerank query", "rerank query 2", "rerank query 3"],
    }


def test_generate_retrieval_prompts_node(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

    class Out(BaseModel):
        x: int | None = None

    class FakePrompts:
        rag_prompts = [
            agent.RagPrompt(rag_prompt="  dense1  "),
            agent.RagPrompt(rag_prompt="  dense2  "),
        ]
        reranker_prompts = [
            agent.RerankPrompt(reranker_prompt="  rerank1  "),
            agent.RerankPrompt(reranker_prompt="  rerank2  "),
        ]

    class FakeStructured:
        def invoke(self, _messages):
            return FakePrompts()

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    out = agent.generate_retrieval_prompts_node(
        {
            "input_query": "user q\nextract x",
            "rag_prompts": [],
            "reranker_prompts": [],
            "rag_context": "",
            "rag_contexts": [],
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _resources(agent, llm=FakeLLM()),
    )
    assert out["rag_prompts"] == ["dense1", "dense2"]
    assert out["reranker_prompts"] == ["rerank1", "rerank2"]


def test_generate_retrieval_prompts_increments_count_after_rewrite(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

    class Out(BaseModel):
        x: int | None = None

    class FakePrompts:
        rag_prompts = [
            agent.RagPrompt(rag_prompt="a"),
            agent.RagPrompt(rag_prompt="b"),
        ]
        reranker_prompts = [
            agent.RerankPrompt(reranker_prompt="r1"),
            agent.RerankPrompt(reranker_prompt="r2"),
        ]

    class FakeStructured:
        def invoke(self, _messages):
            return FakePrompts()

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    out = agent.generate_retrieval_prompts_node(
        {
            "input_query": "q",
            "rag_prompts": ["old_r"],
            "reranker_prompts": ["old_rr"],
            "rag_context": "",
            "rag_contexts": [],
            "answer": "{}",
            "check_decision": "REWRITE",
            "check_reason": "мало данных",
            "rewrite_focus": "нормы",
            "attempt": 0,
        },
        _resources(agent, llm=FakeLLM()),
    )
    assert out["attempt"] == 1
    assert out["rag_prompts"] == ["a", "b"]


def test_rag_search_passes_part_names_to_qdrant(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

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
                payload = {"text": "t1", "parent_text": "parent t1"}

            return [P()]

    monkeypatch.setattr(
        agent,
        "rerank_with_expanded_queries",
        lambda _qs, texts, model, **kwargs: [
            {"index": 0, "text": texts[0], "score": 1.0}
        ],
    )
    monkeypatch.setattr(agent, "get_part_names_for_model", lambda _m: ["АР", "КР"])

    chunks = agent._rag_search_and_rerank(
        _resources(
            agent,
            qdrant_service=FakeQdrant(),
            collection_name="main",
            runtime_cfg=build_runtime_config("on"),
        ),
        ["q1", "q2", "q3"],
        ["rr1", "rr2", "rr3"],
        output_model=Out,
        chapter_module_path="tests.fake_chapter",
    )
    assert chunks == ["t1"]
    assert captured["part_names"] == ["АР", "КР"]
    assert captured["collection_name"] == "main"
    assert captured["limit"] == 50


def test_answer_node_happy_path(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent2

    class Out(BaseModel):
        x: int

    class FakeStructured:
        def invoke(self, _messages):
            return Out(x=1)

    class FakeLLM:
        def with_structured_output(self, _model, strict: bool = True):
            return FakeStructured()

    out = agent2.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "chunks": ["ctx"],
            "chunks_all": ["ctx1", "ctx2"],
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _config(Out),
        _resources(agent2, llm=FakeLLM()),
    )
    assert out["answer"] == Out(x=1).model_dump_json()


def test_answer_node_fallback1_uses_validate_and_dump(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

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

    monkeypatch.setattr(agent, "validate_and_dump_json_str", lambda _m, _s: '{"x": 2}')

    out = agent.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "chunks": ["ctx"],
            "chunks_all": ["ctx1", "ctx2"],
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _config(Out),
        _resources(agent, llm=FakeLLM()),
    )
    assert out["answer"] == '{"x": 2}'


def test_answer_node_fallback2_returns_empty_dict_json(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

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

    out = agent.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "chunks": ["ctx"],
            "chunks_all": ["ctx1", "ctx2"],
            "answer": "",
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _config(Out),
        _resources(agent, llm=FakeLLM()),
    )
    assert out["answer"] == "{}"


def test_answer_node_postprocess_fills_from_merged(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

    class Out(BaseModel):
        x: list
        y: str

    captured_model: dict[str, object] = {}

    class FakeStructured:
        def invoke(self, _messages):
            return Out(x=['5'], y="")

    class FakeLLM:
        def with_structured_output(self, model, strict: bool = True):
            captured_model["model"] = model
            return FakeStructured()

    out = agent.answer_node(
        {
            "input_query": "prompt",
            "rag_prompts": [],
            "reranker_prompts": [],
            "chunks": ["ctx"],
            "chunks_all": ["ctx1", "ctx2"],
            "answer": '{"x":[],"y":"keep"}',
            "fields_to_rewrite": ["x"],
            "verified_fields": ["y"],
            "check_decision": "OK",
            "check_reason": "",
            "rewrite_focus": "",
            "attempt": 0,
        },
        _config(Out),
        _resources(agent, llm=FakeLLM()),
    )
    data = json.loads(out["answer"])
    assert data["x"] == ['5']
    assert data["y"] == "keep"
    assert captured_model["model"].__name__ == "OutPartial"
    assert list(captured_model["model"].model_fields.keys()) == ["x"]


def test_select_generation_model_uses_full_schema_on_first_pass():
    import src.agents.agent as agent

    class Out(BaseModel):
        x: int
        y: str

    assert agent.select_generation_model(Out, None, ["x"]) is Out
    assert agent.select_generation_model(Out, '{"x":1,"y":"a"}', []) is Out


def test_select_generation_model_builds_partial_on_rewrite():
    import src.agents.agent as agent

    class Out(BaseModel):
        x: int
        y: str

    partial = agent.select_generation_model(Out, '{"x":1,"y":"a"}', ["x"])
    assert partial.__name__ == "OutPartial"
    assert list(partial.model_fields.keys()) == ["x"]


def test_route_after_check_limits_rewrites():
    import src.agents.agent as agent

    assert agent.route_after_check({"check_decision": "REWRITE", "attempt": 0}) == "generate_retrieval_prompts_node"
    assert agent.route_after_check({"check_decision": "REWRITE", "attempt": 1}) == "generate_retrieval_prompts_node"
    assert agent.route_after_check({"check_decision": "REWRITE", "attempt": 2}) == "generate_retrieval_prompts_node"
    assert agent.route_after_check({"check_decision": "REWRITE", "attempt": 3}) == agent.END
    assert agent.route_after_check({"check_decision": "OK", "attempt": 0}) == agent.END


def test_init_graph_2_raises_when_collection_unknown_and_no_project_parts_path(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

    class FakeClient:
        def collection_exists(self, _name: str) -> bool:
            return False

    class FakeQdrant:
        def __init__(self):
            self.client = FakeClient()

    monkeypatch.setattr(agent, "build_qdrant_service", lambda x: FakeQdrant())

    with pytest.raises(ValueError, match="project_parts_path не передан"):
        agent.init_graph(collection_name="does-not-exist", project_parts_path=None, runtime_cfg=build_runtime_config('on'))


def test_init_graph_2_existing_collection_does_not_create(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent

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

    monkeypatch.setattr(agent, "build_qdrant_service", lambda x: FakeQdrant())
    monkeypatch.setattr(agent, "create_project_parts", lambda _p: called.__setitem__("parts", called["parts"] + 1))
    monkeypatch.setattr(agent, "create_collection", lambda *_a, **_k: called.__setitem__("create", called["create"] + 1))
    monkeypatch.setattr(agent, "fill_collection", lambda *_a, **_k: called.__setitem__("fill", called["fill"] + 1))
    monkeypatch.setattr(agent, "LlmModel", FakeLlmModel)
    monkeypatch.setattr(agent, "StateGraph", FakeStateGraph)

    compiled, resources = agent.init_graph(collection_name="main", project_parts_path=None, runtime_cfg=build_runtime_config('off'))
    assert compiled == "compiled"
    assert called == {"create": 0, "fill": 0, "parts": 0}
    assert resources.collection_name == "main"
    assert resources.qdrant_service is not None
    assert resources.llm is not None


def test_use_parent(monkeypatch: pytest.MonkeyPatch):
    
    import src.agents.agent as agent
    from src.ecology_chapters.chapter2.models import Geology
    
    @dataclass
    class FakeRetrievedChunk:
        text: str = "dummy"
        parent_text: str= "parent_text"
        score: float = 1.0
        point_id: str | int = 1
        
    def fake_search_by_multi_rag_queries(*args, **kwargs):
        return 'something'
    
    def fake_merge_retrieval_results(x):
        return [FakeRetrievedChunk()]
    
    def fake_rerank_with_expanded_queries(
        reranker_prompts, texts, reranker_model, top_n
    ):
        return [{'index': 0, 'text': 'dummy', 'score': 'score'}]
    
    monkeypatch.setattr(agent, "get_part_names_for_model", lambda x: 'dummy')
    monkeypatch.setattr(agent, "search_by_multi_rag_queries", fake_search_by_multi_rag_queries)
    monkeypatch.setattr(agent, "merge_retrieval_results", fake_merge_retrieval_results)
    monkeypatch.setattr(agent, "chunks_to_texts", lambda chunks: 'dummy')
    monkeypatch.setattr(agent, "rerank_with_expanded_queries", fake_rerank_with_expanded_queries)
    
    out = agent._rag_search_and_rerank(
        resources=_resources(agent, qdrant_service='some',collection_name="ch2_off_parent"),
        rag_prompts=['1', '2'],
        reranker_prompts=['1', '2'],
        output_model=Geology,
        chapter_module_path = "dummy",
    )
    
    assert out == ["parent_text"]


def test_not_use_parent(monkeypatch: pytest.MonkeyPatch):
    import src.agents.agent as agent
    
    class SomeModel(BaseModel):
        pass
    
    @dataclass
    class FakeRetrievedChunk:
        text: str = "dummy"
        parent_text: str = "dummy"
        score: float = 1.0
        point_id: str | int = 1
    
    def fake_search_by_multi_rag_queries(*args, **kwargs):
        return 'dummy'
    
    
    def fake_merge_retrieval_results(x):
        return [FakeRetrievedChunk()]
    
    
    def fake_rerank_with_expanded_queries(
        reranker_prompts, texts, reranker_model, top_n
    ):
        return [{'index': 0, 'text': 'child_text', 'score': 'dummy'}]
    
    
    monkeypatch.setattr(agent, "get_part_names_for_model", lambda x: 'something')
    monkeypatch.setattr(agent, "search_by_multi_rag_queries", fake_search_by_multi_rag_queries)
    monkeypatch.setattr(agent, "merge_retrieval_results", fake_merge_retrieval_results)
    monkeypatch.setattr(agent, "chunks_to_texts", lambda chunks: 'something')
    monkeypatch.setattr(agent, "rerank_with_expanded_queries", fake_rerank_with_expanded_queries)
    
    out = agent._rag_search_and_rerank(
        resources=_resources(agent, qdrant_service='some', collection_name="ch2_off_parent"),
        rag_prompts=['1', '2'],
        reranker_prompts=['1', '2'],
        output_model=SomeModel,
        chapter_module_path="dummy",
    )
    
    assert out == ["child_text"]