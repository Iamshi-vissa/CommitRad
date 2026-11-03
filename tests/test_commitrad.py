"""Tests for CommitRad."""

import commitrad


def test_version() -> None:
    """Test that version is accessible."""
    assert commitrad.__version__ == "0.1.0"


def test_import() -> None:
    """Test that commitrad can be imported."""
    assert commitrad is not None
