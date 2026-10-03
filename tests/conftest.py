from __future__ import annotations

import random

import pytest

from agent_chaos.faults import FaultContext, Surface
from agent_chaos.runner import VirtualClock


@pytest.fixture
def rng():
    return random.Random(1234)


@pytest.fixture
def clock():
    return VirtualClock()


def make_ctx(surface=Surface.REASONING, seed=1, **kw):
    kw.setdefault("rng", random.Random(seed))
    return FaultContext(surface=Surface(surface), **kw)


@pytest.fixture
def ctx_factory():
    return make_ctx
