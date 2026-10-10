from T_mem.llm.embedding_provider import EmbeddingProvider
from T_mem.llm.llm_provider import LLMProvider
from T_mem.llm.reranker_provider import RerankerProvider


def test_models_and_shared_credentials_come_from_environment(monkeypatch):
    monkeypatch.setenv("T_MEM_LLM_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("T_MEM_LLM_API_KEY", "secret")
    monkeypatch.setenv("T_MEM_LLM_BASE_MODEL", "chat-model")
    monkeypatch.setenv("T_MEM_LLM_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("T_MEM_RERANKER_MODEL", "rerank-model")
    monkeypatch.setenv("T_MEM_RERANKER_URL", "https://rerank.test/v1/reranks")

    llm = LLMProvider(max_workers=1)
    embedding = EmbeddingProvider()
    reranker = RerankerProvider()

    assert llm.model == "chat-model"
    assert embedding.model_name == "embedding-model"
    assert embedding._endpoint() == "https://example.test/v1/embeddings"
    assert reranker.model_name == "rerank-model"
    assert reranker._endpoint() == "https://rerank.test/v1/reranks"


def test_legacy_embedding_model_name_still_works(monkeypatch):
    monkeypatch.delenv("T_MEM_LLM_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("T_MEM_EMBEDDING_MODEL", "legacy-embedding")

    assert EmbeddingProvider().model_name == "legacy-embedding"


def test_embedding_provider_splits_large_batches(monkeypatch):
    monkeypatch.setenv("T_MEM_LLM_EMBEDDING_MODEL", "embedding-model")
    monkeypatch.setenv("T_MEM_EMBEDDING_BATCH_SIZE", "2")
    provider = EmbeddingProvider(max_retries=1)
    seen_batches = []

    class Response:
        ok = True
        status_code = 200

        def __init__(self, size):
            self._size = size

        def json(self):
            return {
                "data": [
                    {"index": index, "embedding": [float(index)]}
                    for index in range(self._size)
                ]
            }

    def fake_post(_url, headers, data, timeout):
        import json

        del headers, timeout
        batch = json.loads(data)["input"]
        seen_batches.append(batch)
        return Response(len(batch))

    monkeypatch.setattr(provider.session, "post", fake_post)

    vectors = provider.embed(["a", "b", "c", "d", "e"])

    assert [len(batch) for batch in seen_batches] == [2, 2, 1]
    assert len(vectors) == 5
