"""Declarative experiment profiles for P0-P4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProfileFamily = Literal["keyword", "dense", "registry", "union"]


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    stage: str
    family: ProfileFamily
    description: str
    memory_view_kinds: tuple[str, ...] = ()
    query_view_kinds: tuple[str, ...] = ()
    registry_mode: str = "automatic"
    registry_swap_roles: bool = False
    registry_shuffle_descriptors: bool = False
    union_members: tuple[str, ...] = ()


DEFAULT_PROFILES: dict[str, ProfileSpec] = {
    "p0_keyword": ProfileSpec(
        "p0_keyword", "P0", "keyword", "Raw multilingual BM25-like lexical baseline"
    ),
    "p0_vector": ProfileSpec(
        "p0_vector", "P0", "dense", "Raw dense baseline", ("raw",), ("raw",)
    ),
    "p1_subject": ProfileSpec(
        "p1_subject", "P1", "dense", "Memory subject versus raw query", ("subject",), ("raw",)
    ),
    "p1_rewrite": ProfileSpec(
        "p1_rewrite", "P1", "dense", "Unstructured rewrite control", ("rewrite",), ("raw",)
    ),
    "p1_query_expansion": ProfileSpec(
        "p1_query_expansion",
        "P1",
        "dense",
        "Raw memory versus ordinary query expansion",
        ("raw",),
        ("query_expansion",),
    ),
    "p1_prospective_trigger": ProfileSpec(
        "p1_prospective_trigger",
        "P1",
        "dense",
        "Prospective-trigger flat control",
        ("prospective_trigger",),
        ("raw",),
    ),
    "p2_factor": ProfileSpec(
        "p2_factor",
        "P2",
        "dense",
        "Independent source-factor versus target-requirement matching",
        ("source_factor",),
        ("query_requirement",),
    ),
    "p3_registry_auto": ProfileSpec(
        "p3_registry_auto", "P3", "registry", "Automatic shared-pattern routing"
    ),
    "p3_registry_oracle_m": ProfileSpec(
        "p3_registry_oracle_m",
        "P3",
        "registry",
        "Oracle memory pattern, automatic query pattern",
        registry_mode="oracle_memory",
    ),
    "p3_registry_oracle_q": ProfileSpec(
        "p3_registry_oracle_q",
        "P3",
        "registry",
        "Automatic memory pattern, oracle query pattern",
        registry_mode="oracle_query",
    ),
    "p3_registry_oracle_both": ProfileSpec(
        "p3_registry_oracle_both",
        "P3",
        "registry",
        "Oracle pattern on both sides",
        registry_mode="oracle_both",
    ),
    "p3_registry_swap": ProfileSpec(
        "p3_registry_swap",
        "P3",
        "registry",
        "Source/target role swap ablation",
        registry_swap_roles=True,
    ),
    "p3_registry_shuffle": ProfileSpec(
        "p3_registry_shuffle",
        "P3",
        "registry",
        "Pattern-descriptor shuffle ablation",
        registry_shuffle_descriptors=True,
    ),
    "p4_union": ProfileSpec(
        "p4_union",
        "P4",
        "union",
        "Unweighted rank union for route complementarity",
        union_members=("p0_vector", "p2_factor", "p3_registry_auto"),
    ),
}


def get_profiles(names: list[str] | tuple[str, ...]) -> list[ProfileSpec]:
    unknown = sorted(set(names) - set(DEFAULT_PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {', '.join(unknown)}")
    return [DEFAULT_PROFILES[name] for name in names]

