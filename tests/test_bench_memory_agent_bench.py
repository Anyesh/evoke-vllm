import json
from pathlib import Path

import pytest

from bench.workloads.memory_agent_bench import (
    MabWorkloadConfig,
    build_requests_from_rows,
    exact_match_score,
    extract_answer,
    filter_rows_by_source,
    metric_max_over_ground_truths,
    normalize_answer,
    select_examples,
    substring_exact_match_score,
    truncate_to_token_budget,
)

FIXTURES = Path(__file__).parent / "bench_fixtures"
ROWS = json.loads((FIXTURES / "mab_rows.json").read_text())


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The Rayleigh Scattering!") == "rayleigh scattering"
    assert normalize_answer("a cat, an owl, the dog") == "cat owl dog"


def test_exact_match_score_ignores_case_and_articles():
    assert exact_match_score("The Paris", "paris") is True
    assert exact_match_score("Paris", "the Paris") is True
    assert exact_match_score("Berlin", "Paris") is False


def test_substring_exact_match_score():
    assert substring_exact_match_score("It happened in Paris, France.", "Paris") is True
    assert substring_exact_match_score("Nowhere relevant", "Paris") is False


def test_metric_max_over_ground_truths_flat_list():
    assert metric_max_over_ground_truths(
        exact_match_score, "paris", ["london", "Paris"]
    )
    assert not metric_max_over_ground_truths(
        exact_match_score, "berlin", ["london", "Paris"]
    )


def test_metric_max_over_ground_truths_nested_list():
    nested = [["Rayleigh scattering", "rayleigh scattering"]]
    assert metric_max_over_ground_truths(
        exact_match_score, "Rayleigh Scattering", nested
    )


def test_metric_max_over_ground_truths_empty_is_false():
    assert metric_max_over_ground_truths(exact_match_score, "anything", []) is False


def test_filter_rows_by_source():
    filtered = filter_rows_by_source(ROWS, ("ruler_qa1", "event_qa"))
    sources = {row["metadata"]["source"] for row in filtered}
    assert sources == {"ruler_qa1", "event_qa"}
    assert len(filtered) == 2


def test_select_examples_returns_all_when_fewer_than_n():
    selected = select_examples(ROWS, n=10, seed=1)
    assert selected == ROWS


def test_select_examples_is_deterministic_for_a_fixed_seed():
    a = select_examples(ROWS, n=2, seed=42)
    b = select_examples(ROWS, n=2, seed=42)
    assert a == b
    assert len(a) == 2


def test_build_requests_from_rows_produces_one_request_per_question():
    config = MabWorkloadConfig(
        workload_id="W1",
        track="Accurate_Retrieval",
        sources=("ruler_qa1", "event_qa"),
        scorer="substring_exact_match",
        n_examples=8,
        seed=42,
        max_tokens=32,
    )
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)

    assert workload.workload_id == "W1"
    assert len(workload.requests) == 3
    first = workload.requests[0]
    assert first.messages[0]["role"] == "system"
    assert "Why is the sky blue?" in first.messages[1]["content"]
    assert first.max_tokens == 32
    assert first.temperature == 0.0
    assert first.kv_transfer_params["evoke"]["source_type"] == "user"
    assert first.ground_truths == ["Rayleigh scattering", "rayleigh scattering"]


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Out:
            def __init__(self, ids):
                self.input_ids = ids

        return Out(text.split())

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def _budget_config(**overrides):
    base = dict(
        workload_id="W1",
        track="Accurate_Retrieval",
        sources=("ruler_qa1", "event_qa"),
        scorer="substring_exact_match",
        n_examples=8,
        seed=42,
        max_tokens=32,
    )
    base.update(overrides)
    return MabWorkloadConfig(**base)


def test_truncate_to_token_budget_cuts_front_keep():
    tok = WordTokenizer()
    text = "a b c d e f"
    assert truncate_to_token_budget(text, 3, tok) == "a b c"
    assert truncate_to_token_budget(text, 10, tok) == text


def test_build_requests_truncates_context_but_keeps_question():
    # The native MemoryAgentBench contexts (65k to 421k tokens by source
    # name) cannot be served inside the profiles' max_model_len, so the
    # loader must bound the context deterministically while leaving the
    # question intact; arms compare against identical truncated inputs.
    config = _budget_config(
        context_token_budget=4, tokenizer_id="stub", max_questions_per_example=None
    )
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered, tokenizer=WordTokenizer())
    for request in workload.requests:
        user = request.messages[1]["content"]
        context_part, question_part = user.split("\n\nQuestion: ")
        assert len(context_part.split()) <= 4
        assert question_part


def test_build_requests_caps_questions_per_example():
    config = _budget_config(max_questions_per_example=1)
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)
    per_example: dict[str, int] = {}
    for request in workload.requests:
        key = request.kv_transfer_params["evoke"]["evoke_session"]
        per_example[key] = per_example.get(key, 0) + 1
    assert per_example
    assert all(count == 1 for count in per_example.values())


def test_context_budget_requires_tokenizer():
    with pytest.raises(ValueError):
        _budget_config(context_token_budget=4, tokenizer_id=None)

    config = _budget_config(context_token_budget=4, tokenizer_id="stub")
    filtered = filter_rows_by_source(ROWS, config.sources)
    with pytest.raises(ValueError):
        build_requests_from_rows(config, filtered, tokenizer=None)


def test_scoring_uses_configured_scorer():
    config = MabWorkloadConfig(
        workload_id="W1",
        track="Accurate_Retrieval",
        sources=("ruler_qa1",),
        scorer="substring_exact_match",
        n_examples=8,
        seed=42,
    )
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)
    request = workload.requests[0]

    assert workload.score(request, "The answer is Rayleigh scattering, yes.") == 1.0
    assert workload.score(request, "no idea") == 0.0


def test_extract_answer_from_valid_json():
    completion = '{"answer": "C. The Brandt couple", "reasoning": "the will"}'
    assert extract_answer(completion) == "C. The Brandt couple"


def test_extract_answer_from_truncated_json():
    # The observed W2 completion shape: max_tokens cuts the JSON somewhere
    # inside "reasoning", so the document never closes and json.loads alone
    # cannot recover the answer field.
    completion = '{"answer":"A. Sheila Webb", "reasoning":"Sheila Webb was the'
    assert extract_answer(completion) == "A. Sheila Webb"


def test_extract_answer_passthrough_without_answer_field():
    assert extract_answer("Rayleigh scattering.") == "Rayleigh scattering."
    assert extract_answer('{"verdict": "yes"}') == '{"verdict": "yes"}'


def test_scoring_extracts_json_answer_before_matching():
    config = MabWorkloadConfig(
        workload_id="W2",
        track="Long_Range_Understanding",
        sources=("detectiveQA",),
        scorer="exact_match",
        n_examples=8,
        seed=42,
    )
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)
    request = workload.requests[0]
    assert request.ground_truths == ["Holmes", "Detective Holmes"]

    assert workload.score(request, '{"answer":"Holmes", "reasoning":"the') == 1.0
    assert workload.score(request, '{"answer":"Watson", "reasoning":"the') == 0.0


def test_example_major_is_the_default_issue_order():
    config = _budget_config()
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)
    ids = [request.request_id for request in workload.requests]
    assert ids == ["W1-ex0-q0", "W1-ex0-q1", "W1-ex1-q0"]


def test_round_robin_interleaves_questions_across_examples():
    config = _budget_config(issue_order="round_robin")
    filtered = filter_rows_by_source(ROWS, config.sources)
    workload = build_requests_from_rows(config, filtered)
    ids = [request.request_id for request in workload.requests]
    assert ids == ["W1-ex0-q0", "W1-ex1-q0", "W1-ex0-q1"]


def test_unknown_issue_order_rejected():
    with pytest.raises(ValueError):
        _budget_config(issue_order="shuffled")


def _synthetic_rows(n_examples: int, n_questions: int) -> list[dict]:
    return [
        {
            "context": f"context for example {e}",
            "questions": [f"question {e}-{q}" for q in range(n_questions)],
            "answers": [[f"answer {e}-{q}"] for q in range(n_questions)],
            "metadata": {"source": "synthetic"},
        }
        for e in range(n_examples)
    ]


def test_hot_cold_pins_hot_revisits_between_cold_scans():
    config = MabWorkloadConfig(
        workload_id="W1S",
        track="Accurate_Retrieval",
        sources=("synthetic",),
        scorer="substring_exact_match",
        n_examples=4,
        seed=42,
        issue_order="hot_cold",
        hot_examples=1,
        cold_questions_per_example=2,
    )
    workload = build_requests_from_rows(config, _synthetic_rows(4, 3))
    ids = [request.request_id for request in workload.requests]
    assert ids == [
        "W1S-ex0-q0",
        "W1S-ex1-q0",
        "W1S-ex2-q0",
        "W1S-ex0-q1",
        "W1S-ex3-q0",
        "W1S-ex1-q1",
        "W1S-ex0-q2",
        "W1S-ex2-q1",
        "W1S-ex3-q1",
    ]


def test_hot_cold_keeps_every_hot_question_and_caps_cold():
    config = MabWorkloadConfig(
        workload_id="W1S",
        track="Accurate_Retrieval",
        sources=("synthetic",),
        scorer="substring_exact_match",
        n_examples=8,
        seed=42,
        max_questions_per_example=12,
        issue_order="hot_cold",
        hot_examples=2,
        cold_questions_per_example=6,
    )
    workload = build_requests_from_rows(config, _synthetic_rows(8, 12))
    per_example: dict[int, int] = {}
    for request in workload.requests:
        e = request.metadata["example_index"]
        per_example[e] = per_example.get(e, 0) + 1
    assert per_example[0] == 12
    assert per_example[1] == 12
    assert all(per_example[e] == 6 for e in range(2, 8))
    assert len(workload.requests) == 60


def test_aggregate_score_ignores_none_and_averages():
    from bench.workloads.base import Workload

    assert Workload.aggregate([1.0, 0.0, None, 1.0]) == pytest.approx(2 / 3)
    assert Workload.aggregate([None, None]) is None


def test_unknown_scorer_rejected():
    with pytest.raises(ValueError):
        MabWorkloadConfig(
            workload_id="W1",
            track="Accurate_Retrieval",
            sources=("ruler_qa1",),
            scorer="not_a_real_scorer",
        )
