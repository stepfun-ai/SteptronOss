import pytest

from steptronoss.foo import foo

pytestmark = pytest.mark.cpu


def test_foo():
    assert foo("foo") == "foo"
