import pytest

from eval.rec_eval import evaluate


def test_synthetic_evaluation_is_deterministic_and_reports_cutoff_metrics():
    first = evaluate(sessions=100, k=10, seed=17)
    second = evaluate(sessions=100, k=10, seed=17)
    assert first == second
    assert first["sessions"] == 100
    assert 0 <= first["mrr_at_k"] <= 1
    assert 0 <= first["ndcg_at_k"] <= 1
    assert 0 <= first["precision_at_k"] <= 1


@pytest.mark.parametrize("sessions,k", [(0, 10), (10, 0), (-1, 5)])
def test_evaluation_rejects_nonpositive_dimensions(sessions, k):
    with pytest.raises(ValueError):
        evaluate(sessions=sessions, k=k)
