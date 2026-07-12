"""Reader for verdant's content-addressed blob store.

Mirrors the store's on-disk layout: a payload for a
64-char lowercase blake3 digest lives at ``<root>/<digest[:2]>/<digest>.payload``,
sharded by the first two hex chars. When ``root`` is ``None``, or a digest is
absent from the store (evicted, or the store was never populated for a given
trace), ``resolve`` falls back to deterministic filler bytes so a replay stays
byte-reproducible across runs without the original text ever having to exist
on this machine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import blake3

ZERO_DIGEST = "0" * 64

HIT = "hit"
MISS = "miss"
CORRUPT = "corrupt"


@dataclass(frozen=True)
class Resolution:
    data: bytes
    status: str


class CasReader:
    def __init__(self, root: Path | str | None, verify_integrity: bool = True) -> None:
        self.root = Path(root) if root is not None else None
        self.verify_integrity = verify_integrity

    def payload_path(self, digest_hex: str) -> Path:
        if self.root is None:
            raise ValueError("payload_path requires a configured CAS root")
        return self.root / digest_hex[:2] / f"{digest_hex}.payload"

    def resolve(self, digest_hex: str, fallback_length: int) -> Resolution:
        if self.root is not None and len(digest_hex) >= 2:
            path = self.payload_path(digest_hex)
            if path.exists():
                data = path.read_bytes()
                if (
                    self.verify_integrity
                    and blake3.blake3(data).hexdigest() != digest_hex
                ):
                    return Resolution(
                        data=filler_bytes(digest_hex, fallback_length), status=CORRUPT
                    )
                return Resolution(data=data, status=HIT)
        return Resolution(data=filler_bytes(digest_hex, fallback_length), status=MISS)


def filler_bytes(digest_hex: str, length: int) -> bytes:
    """Deterministic filler, exact length, same digest always yields the same bytes.

    Determinism matters here: two replayed calls that share a segment digest
    must produce byte-identical filler so the server-side prefix hash still
    matches between them, which is the whole point of replaying a growing
    prefix at temperature=0 even when the original text is unrecoverable.
    """
    if length <= 0:
        return b""
    seed = hashlib.sha256(digest_hex.encode("ascii")).digest()
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(
            hashlib.sha256(seed + counter.to_bytes(4, "big"))
            .hexdigest()
            .encode("ascii")
        )
        counter += 1
    return bytes(out[:length])
