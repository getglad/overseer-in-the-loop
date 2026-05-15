"""Smoke test: verifies the toolchain runs end-to-end at the post-01 baseline."""


def test_python_works() -> None:
    """Sanity check — fails loudly if the test runner itself is broken."""
    assert 1 + 1 == 2
