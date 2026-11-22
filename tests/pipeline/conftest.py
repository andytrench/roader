"""Shared synthetic scene for the pipeline lane (stages 3 to 9)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roader.io import NOISE_CLASSES, VectorEdge, from_synthetic
from roader.model.synthetic import overpass_model, overpass_points


def synthetic_vectors(model, seed: int = 3, bias: float = 1.5, noise: float = 0.3, step: float = 10.0) -> list[VectorEdge]:
    """Centerlines of the truth roads with a slowly varying lateral bias and vertex noise (spec section 3 defects)."""
    rng = np.random.default_rng(seed)
    out = []
    for road in model.roads:
        s = np.arange(0.0, road.length + 1e-9, step)
        xy = np.array([road.alignment.offset_point(float(v), 0.0) for v in s])
        h = road.alignment.evaluate(0.0)[2]
        normal = np.array([-math.sin(h), math.cos(h)])
        offset = bias * np.sin(s / 120.0) + rng.normal(0.0, noise, len(s))
        ref = road.vector_refs[0]
        tags = dict(ref.tags)
        if road.oneway:
            tags["oneway"] = "yes"
        if road.kind == "crossing_street":
            tags["name"] = road.name
        else:
            tags["ref"] = "I 99"
        out.append(VectorEdge("synthetic", ref.feature_id, xy + offset[:, None] * normal, tags))
    return out


@pytest.fixture(scope="session")
def truth():
    return overpass_model()


@pytest.fixture(scope="session")
def synthetic_points(truth):
    return overpass_points(truth)


@pytest.fixture(scope="session")
def points(synthetic_points):
    return from_synthetic(synthetic_points)


@pytest.fixture(scope="session")
def labels(synthetic_points):
    return synthetic_points.truth[~np.isin(synthetic_points.classification, NOISE_CLASSES)]


@pytest.fixture(scope="session")
def vectors(truth):
    return synthetic_vectors(truth)


@pytest.fixture(scope="session")
def result(points, vectors, truth):
    from roader.pipeline import run_pipeline

    return run_pipeline(points, vectors, truth.frame, truth.unit)


@pytest.fixture(scope="session")
def model(result):
    return result.model


def by_feature(model, feature_id: str):
    for road in model.roads:
        if any(ref.feature_id == feature_id for ref in road.vector_refs):
            return road
    raise KeyError(feature_id)
