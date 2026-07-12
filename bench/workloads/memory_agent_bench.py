"""MemoryAgentBench workloads (W1 accurate-retrieval, W2 long-range).

Consumes the HF dataset directly rather than the upstream bash harness:
that harness wires in agentic memory
frameworks (cognee, letta, hipporag) with conflicting pins, for a "long
context agent" path this package only needs to concatenate ``context`` and
one ``question`` per request. ``normalize_answer``, ``exact_match_score``,
and ``substring_exact_match_score`` below are ported line-for-line from
``utils/eval_other_utils.py`` in github.com/HUST-AI-HYZ/MemoryAgentBench
(MIT license, per the dataset card and repository LICENSE), so scored
results are comparable with the upstream benchmark's own numbers.

Dataset: ``ai-hyz/MemoryAgentBench`` on the Hugging Face Hub, MIT-licensed.
``DATASET_REVISION`` pins the exact commit so a scored run is reproducible
even if the dataset is updated later; refresh it deliberately by checking
``https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench``.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import zip_longest

import datasets
import transformers

from bench.workloads.base import ChatRequest, Workload

HF_DATASET_NAME = "ai-hyz/MemoryAgentBench"
DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"

TRACK_ACCURATE_RETRIEVAL = "Accurate_Retrieval"
TRACK_LONG_RANGE_UNDERSTANDING = "Long_Range_Understanding"

SCORER_SUBSTRING_EXACT_MATCH = "substring_exact_match"
SCORER_EXACT_MATCH = "exact_match"

ISSUE_ORDER_EXAMPLE_MAJOR = "example_major"
ISSUE_ORDER_ROUND_ROBIN = "round_robin"
ISSUE_ORDER_HOT_COLD = "hot_cold"
_ISSUE_ORDERS = (
    ISSUE_ORDER_EXAMPLE_MAJOR,
    ISSUE_ORDER_ROUND_ROBIN,
    ISSUE_ORDER_HOT_COLD,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. Answer as briefly as possible."
)


class WorkloadDataUnavailable(RuntimeError):
    """Raised when the HF dataset cannot be fetched (offline, no cache).

    The message always names the prefetch subcommand so a run failing on a
    disconnected box tells the operator exactly what to do next, instead of
    letting a raw ``datasets``/``huggingface_hub`` traceback surface.
    """


def normalize_answer(answer_text: str) -> str:
    text = answer_text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


_ANSWER_FIELD = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)')


def extract_answer(completion: str) -> str:
    """Pull the ``answer`` field out of a JSON-shaped completion.

    detective_qa questions instruct the model to answer as JSON with
    ``answer`` and ``reasoning`` keys, and ``max_tokens`` usually cuts the
    document off inside ``reasoning``, so a strict parse alone cannot
    recover the answer; the regex accepts an unterminated document.
    Upstream MemoryAgentBench likewise scores an extracted answer portion
    (``utils/eval_other_utils.py::parse_output``), not the raw output, so
    extraction preserves scorer parity. Completions with no ``answer``
    field pass through unchanged.
    """
    try:
        parsed = json.loads(completion)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "answer" in parsed:
        return str(parsed["answer"])
    match = _ANSWER_FIELD.search(completion)
    if match:
        return re.sub(r"\\(.)", r"\1", match.group(1))
    return completion


def metric_max_over_ground_truths(
    metric_function, prediction: str, ground_truths: Sequence[str]
) -> bool:
    if isinstance(ground_truths, str):
        ground_truth_list: list[str] = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        ground_truth_list = [gt for sublist in ground_truths for gt in sublist]
    else:
        ground_truth_list = list(ground_truths)
    if not ground_truth_list:
        return False
    return max(metric_function(prediction, gt) for gt in ground_truth_list)


_SCORERS = {
    SCORER_SUBSTRING_EXACT_MATCH: substring_exact_match_score,
    SCORER_EXACT_MATCH: exact_match_score,
}


@dataclass(frozen=True)
class MabWorkloadConfig:
    workload_id: str
    track: str
    sources: tuple[str, ...]
    scorer: str
    n_examples: int = 8
    seed: int = 42
    max_tokens: int = 64
    # The dataset's native contexts run 65k to 421k tokens (the source
    # names carry the size: eventqa_65536, ruler_qa1_197K, ...), which no
    # 16GB-card serving window can hold, so contexts are cut to this many
    # tokens before message composition. Truncation is deterministic and
    # identical across benchmark arms; absolute scores are NOT comparable
    # to the upstream leaderboard once it is set, only arm-relative parity
    # is meaningful. None disables truncation.
    context_token_budget: int | None = None
    # Deterministic first-N cap per example to bound a cell's wall clock
    # (W1 otherwise issues 800 requests per cell). None keeps every question.
    max_questions_per_example: int | None = None
    tokenizer_id: str | None = None
    # example_major issues every question of an example before moving on,
    # so a context stays GPU-resident across its own questions and is never
    # revisited after eviction: zero offload restores at any GPU pressure
    # (the vacuous 2026-07-11 matrix, 100k connector queries, 0 hits).
    # round_robin cycles examples by question index so each revisit lands
    # after the other examples' contexts have flowed through the pool,
    # deterministically and identically for every arm. hot_cold splits
    # examples into a hot set revisited every round and cold scan traffic
    # spread between the revisits: uniform re-access gives a
    # reuse-frequency policy nothing to exploit, so this is the access
    # pattern that separates scored eviction from plain LRU (or shows it
    # cannot be separated).
    issue_order: str = ISSUE_ORDER_EXAMPLE_MAJOR
    # hot_cold only: the first N selected examples keep every question
    # (one per round); the rest are capped at cold_questions_per_example
    # and interleaved round-robin so consecutive cold slots are distinct
    # contexts, which is what makes the scan actually flush an LRU pool.
    hot_examples: int = 2
    cold_questions_per_example: int | None = None

    def __post_init__(self) -> None:
        if self.scorer not in _SCORERS:
            raise ValueError(
                f"unknown scorer {self.scorer!r}, want one of {sorted(_SCORERS)}"
            )
        if self.issue_order not in _ISSUE_ORDERS:
            raise ValueError(
                f"unknown issue_order {self.issue_order!r}, want one of {_ISSUE_ORDERS}"
            )
        if self.context_token_budget is not None and self.tokenizer_id is None:
            raise ValueError(
                "context_token_budget requires tokenizer_id so the cut is "
                "made with the same tokenizer the served model uses"
            )


def filter_rows_by_source(rows, sources: tuple[str, ...]) -> list[dict]:
    wanted = set(sources)
    return [row for row in rows if row.get("metadata", {}).get("source") in wanted]


def fetch_rows(config: MabWorkloadConfig) -> list[dict]:
    try:
        raw = datasets.load_dataset(
            HF_DATASET_NAME, split=config.track, revision=DATASET_REVISION
        )
    except Exception as exc:
        raise WorkloadDataUnavailable(
            f"could not load {HF_DATASET_NAME}:{config.track}@{DATASET_REVISION} "
            f"({exc}). Run `python -m bench prefetch --workload "
            f"{config.workload_id}` on a connected machine first, or check "
            "the local HF cache (HF_HOME)."
        ) from exc
    return filter_rows_by_source(raw, config.sources)


def select_examples(rows: list[dict], n: int, seed: int) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in indices]


def _as_ground_truths(raw_answer: object) -> list[str]:
    if isinstance(raw_answer, str):
        return [raw_answer]
    if isinstance(raw_answer, list):
        flat: list[str] = []
        for item in raw_answer:
            if isinstance(item, list):
                flat.extend(str(x) for x in item)
            else:
                flat.append(str(item))
        return flat
    return [str(raw_answer)]


def truncate_to_token_budget(text: str, budget: int, tokenizer) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def build_requests_from_rows(
    config: MabWorkloadConfig, rows: list[dict], tokenizer=None
) -> Workload:
    if config.context_token_budget is not None and tokenizer is None:
        raise ValueError(
            "config sets context_token_budget but no tokenizer was provided"
        )
    selected = select_examples(rows, config.n_examples, config.seed)
    scorer_fn = _SCORERS[config.scorer]
    per_example: list[list[ChatRequest]] = []
    for example_index, row in enumerate(selected):
        context = row["context"]
        if config.context_token_budget is not None:
            context = truncate_to_token_budget(
                context, config.context_token_budget, tokenizer
            )
        questions = row["questions"]
        if config.max_questions_per_example is not None:
            questions = questions[: config.max_questions_per_example]
        answers = row["answers"]
        example_id = f"{config.workload_id}-ex{example_index}"
        example_requests: list[ChatRequest] = []
        for q_index, question in enumerate(questions):
            ground_truths = (
                _as_ground_truths(answers[q_index]) if q_index < len(answers) else []
            )
            example_requests.append(
                ChatRequest(
                    request_id=f"{example_id}-q{q_index}",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"{context}\n\nQuestion: {question}",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=config.max_tokens,
                    kv_transfer_params={
                        "evoke": {
                            "evoke_session": example_id,
                            "source_type": "user",
                            "priority": 1.0,
                        }
                    },
                    ground_truths=ground_truths,
                    metadata={
                        "example_index": example_index,
                        "question_index": q_index,
                        "source": row.get(
                            "source", row.get("metadata", {}).get("source", "")
                        ),
                    },
                )
            )
        per_example.append(example_requests)

    if config.issue_order == ISSUE_ORDER_ROUND_ROBIN:
        requests = [
            request
            for round_of_questions in zip_longest(*per_example)
            for request in round_of_questions
            if request is not None
        ]
    elif config.issue_order == ISSUE_ORDER_HOT_COLD:
        hot = per_example[: config.hot_examples]
        cold = per_example[config.hot_examples :]
        if config.cold_questions_per_example is not None:
            cold = [example[: config.cold_questions_per_example] for example in cold]
        # Round-robin the cold queue so consecutive scan slots are distinct
        # contexts; a run of questions from one cold example would be served
        # by the GPU prefix cache and never touch the offload pool.
        cold_queue = [
            request
            for round_of_questions in zip_longest(*cold)
            for request in round_of_questions
            if request is not None
        ]
        rounds = max((len(example) for example in hot), default=0)
        per_round = math.ceil(len(cold_queue) / rounds) if rounds else 0
        requests = []
        cold_index = 0
        for round_index in range(rounds):
            for example in hot:
                if round_index < len(example):
                    requests.append(example[round_index])
            requests.extend(cold_queue[cold_index : cold_index + per_round])
            cold_index += per_round
        requests.extend(cold_queue[cold_index:])
    else:
        requests = [request for example in per_example for request in example]

    def score_fn(request: ChatRequest, completion: str) -> float | None:
        if not request.ground_truths:
            return None
        matched = metric_max_over_ground_truths(
            scorer_fn, extract_answer(completion), request.ground_truths
        )
        return 1.0 if matched else 0.0

    return Workload(
        workload_id=config.workload_id,
        requests=requests,
        score_fn=score_fn,
        stats={"n_examples": len(selected), "n_requests": len(requests)},
    )


def load_workload(config: MabWorkloadConfig) -> Workload:
    rows = fetch_rows(config)
    tokenizer = None
    if config.context_token_budget is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer_id)
    return build_requests_from_rows(config, rows, tokenizer=tokenizer)
