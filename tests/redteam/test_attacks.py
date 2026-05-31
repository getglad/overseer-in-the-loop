"""Structural integrity of the attack corpus — runs offline, no gate calls."""

from __future__ import annotations

import dataclasses

import pytest

from src.redteam.attacks import (
    CORPUS,
    LAYER_GUARDRAIL_AGENT,
    LAYER_RULES,
    Attack,
    Expected,
)

_VALID_LAYERS = {None, LAYER_RULES, LAYER_GUARDRAIL_AGENT}
_EXPECTED_CATEGORIES = {
    "A1", "A2", "A3", "A4", "A5", "A6", "A7",
    "prompt-injection", "obfuscation", "gap", "benign-control",
}


class TestCorpusIntegrity:
    """Every corpus entry is well-formed and the set covers the threat model."""

    def test_corpus_is_non_trivial(self) -> None:
        """A red-team corpus must be a comprehensive battery, not a token sample."""
        assert len(CORPUS) >= 40

    def test_attack_ids_unique(self) -> None:
        """attack_id is the scorecard key, so duplicates would silently merge rows."""
        ids = [a.attack_id for a in CORPUS]
        assert len(ids) == len(set(ids))

    def test_every_attack_has_required_fields(self) -> None:
        """No attack may ship with an empty id, category, description, or tool name."""
        for attack in CORPUS:
            assert attack.attack_id
            assert attack.category
            assert attack.description
            assert attack.tool_name
            assert isinstance(attack.tool_args, dict)
            assert isinstance(attack.expected, Expected)

    def test_expected_layer_values_are_valid(self) -> None:
        """expected_layer must be a known tier or None — a typo would never match."""
        for attack in CORPUS:
            assert attack.expected_layer in _VALID_LAYERS

    def test_is_deterministic_tracks_rules_layer(self) -> None:
        """An attack is CI-deterministic exactly when the rules tier decides it."""
        for attack in CORPUS:
            assert attack.is_deterministic == (attack.expected_layer == LAYER_RULES)

    def test_all_categories_covered(self) -> None:
        """The battery spans the full A1-A7 taxonomy plus the cross-cutting families."""
        present = {a.category for a in CORPUS}
        assert present == _EXPECTED_CATEGORIES

    def test_has_benign_controls(self) -> None:
        """Controls prove the gate discriminates — a paranoid gate fails these."""
        controls = [a for a in CORPUS if a.expected == Expected.ALLOW]
        assert len(controls) >= 5

    def test_attack_is_frozen(self) -> None:
        """Attack records are immutable so the shared corpus can't be mutated mid-run."""
        attack = CORPUS[0]
        assert isinstance(attack, Attack)
        with pytest.raises(dataclasses.FrozenInstanceError):
            attack.attack_id = "mutated"  # type: ignore[misc]
