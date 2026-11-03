"""Tests for CommitRad."""

import commitrad


def test_version():
    """Test that version is accessible."""
    assert commitrad.__version__ == "0.1.0"


def test_import():
    """Test that commitrad can be imported."""
    assert commitrad is not None