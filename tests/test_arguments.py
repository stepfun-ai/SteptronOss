import sys

import pytest

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _clear_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])


def _parse(opts):
    sys.argv = ["prog"] + opts
    from steptronoss.utils.arguments import parse_args

    return parse_args()


def test_normal_key_value():
    result = _parse(["lr=0.001"])
    assert result == {"lr": 0.001}


def test_split_with_equals_in_value():
    result = _parse(["load_path=s3://bucket/path?key=val"])
    assert result == {"load_path": "s3://bucket/path?key=val"}


def test_multiple_equals_in_value():
    result = _parse(["envs=A=1,B=2"])
    assert result == {"envs": "A=1,B=2"}


def test_integer_and_bool_literals():
    result = _parse(["train_iters=100", "flag=True"])
    assert result == {"train_iters": 100, "flag": True}


def test_false_string_stays_as_string():
    result = _parse(["flag=false"])
    assert result["flag"] == "false"


def test_none_string_stays_as_string():
    result = _parse(["val=null"])
    assert result["val"] == "null"


def test_opts_without_equals_are_skipped():
    result = _parse(["no_equals_token", "lr=0.1"])
    assert result == {"lr": 0.1}
