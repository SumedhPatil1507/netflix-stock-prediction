from src.recommendations.engine import RecommendationEngine


class FakeTower:
    def user_vector(self, _user_id):
        return [1.0, 0.0]

    def title_vector(self, _content_id):
        return None


class FakeCandidates:
    def __init__(self, items):
        self.items = items

    def retrieve(self, _vector, **_kwargs):
        return self.items

    def by_ids(self, _content_ids):
        return []


class FakeRanker:
    def score(self, item, **_kwargs):
        return item["retrieval_score"], 0.8


def test_recommendations_return_explainable_contract_and_contextual_rows():
    engine = RecommendationEngine(FakeTower(), FakeRanker(), FakeCandidates([
        {"content_id": "thriller-1", "title": "Night Signal", "genres": ["Thriller"],
         "duration_seconds": 1500, "retrieval_score": 0.942},
        {"content_id": "drama-1", "title": "Long Road", "genres": ["Drama"],
         "duration_seconds": 5400, "retrieval_score": 0.8},
    ]))

    result = engine.recommend("user-1", events=[], context={"hour": 23, "weekend": True})

    assert set(result) == {"verdict", "reasoning", "recommendation", "next_steps"}
    assert result["verdict"] == "Top Picks for You: Night Signal"
    assert "94.2%" in result["reasoning"]
    assert "late-night" in result["reasoning"]
    assert "suggest this short title" in result["recommendation"]
    assert len(result["next_steps"]) == 3


def test_empty_candidate_set_still_returns_the_same_contract():
    engine = RecommendationEngine(FakeTower(), FakeRanker(), FakeCandidates([]))
    result = engine.recommend("user-1", events=[], context={})

    assert set(result) == {"verdict", "reasoning", "recommendation", "next_steps"}
    assert result["verdict"] == "No matching title"
    assert "unavailable" in result["reasoning"]
