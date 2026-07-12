"""Relevance-driven CPU KV-cache offload policy for stock vLLM.

Submodules that touch vLLM (spec, manager, policy) are intentionally not
imported here, so that ``evoke_vllm.config`` stays importable in
environments without vLLM installed.
"""

__version__ = "0.1.1"
