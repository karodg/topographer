"""
Unit tests for TopoGrapher Command-Line Interface (CLI).
"""

import sys
import os
import subprocess
from unittest.mock import patch
import pytest

from topographer.cli import main


def test_cli_version(capsys):
    with patch.object(sys, "argv", ["topographer", "version"]):
        main()
    captured = capsys.readouterr()
    assert "TopoGrapher v1.0.0" in captured.out
    assert "PyTorch Version" in captured.out


def test_cli_help(capsys):
    with patch.object(sys, "argv", ["topographer", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "extract" in captured.out
    assert "benchmark" in captured.out
    assert "route" in captured.out
