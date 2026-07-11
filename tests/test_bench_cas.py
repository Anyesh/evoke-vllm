from pathlib import Path

from bench.workloads.cas import CasReader, filler_bytes

FIXTURES = Path(__file__).parent / "bench_fixtures" / "cas"

HIT_DIGEST = "bf4925425a1fb17d7fed8f06598a65e89721a7512c19fd4a35c696770fe27cc2"
HIT_CONTENT = b"hello from the fake CAS store"
CORRUPT_DIGEST = "95a7e7b2cf6228b49d72c5cc02dcb1f0866649b37b1e07b89c5ea2d79716e2df"
MISS_DIGEST = "c26a601c8a64d54cd608bad35191415780a5e6068a2530e8ced05b6910f1b8c9"


def test_hit_returns_real_bytes():
    reader = CasReader(FIXTURES)
    resolution = reader.resolve(HIT_DIGEST, fallback_length=999)
    assert resolution.status == "hit"
    assert resolution.data == HIT_CONTENT


def test_miss_returns_filler_of_requested_length():
    reader = CasReader(FIXTURES)
    resolution = reader.resolve(MISS_DIGEST, fallback_length=40)
    assert resolution.status == "miss"
    assert len(resolution.data) == 40


def test_none_root_always_misses():
    reader = CasReader(None)
    resolution = reader.resolve(HIT_DIGEST, fallback_length=10)
    assert resolution.status == "miss"
    assert len(resolution.data) == 10


def test_corrupt_payload_falls_back_to_filler():
    reader = CasReader(FIXTURES, verify_integrity=True)
    resolution = reader.resolve(CORRUPT_DIGEST, fallback_length=20)
    assert resolution.status == "corrupt"
    assert len(resolution.data) == 20


def test_corrupt_payload_returned_as_is_when_integrity_disabled():
    reader = CasReader(FIXTURES, verify_integrity=False)
    resolution = reader.resolve(CORRUPT_DIGEST, fallback_length=20)
    assert resolution.status == "hit"


def test_filler_is_deterministic_and_length_exact():
    a = filler_bytes(MISS_DIGEST, 137)
    b = filler_bytes(MISS_DIGEST, 137)
    assert a == b
    assert len(a) == 137


def test_filler_differs_by_digest():
    a = filler_bytes(HIT_DIGEST, 64)
    b = filler_bytes(MISS_DIGEST, 64)
    assert a != b


def test_filler_zero_length_is_empty():
    assert filler_bytes(MISS_DIGEST, 0) == b""
