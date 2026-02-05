import os
import sys
import types

import pytest

from steptronoss.exp.base_exp import BaseExp

pytestmark = pytest.mark.cpu


class DummyExp(BaseExp):
    pass


class DummyExpWithMerge(DummyExp):
    _allow_set_new_attr = True

    def __init__(self):
        super().__init__()
        self.merge_args = None
        self.merge_kwargs = None

    def merge(self, args, exists_only=False):
        self.merge_args = args
        self.merge_kwargs = {"exists_only": exists_only}
        return [("x", "old", "new")]


def test_exp_name_uses_filename_and_suffix():
    exp = DummyExp()
    exp.suffix = "unit"

    base_name = os.path.basename(__file__).split(".")[0]
    exp_name = str(exp.exp_name)
    assert exp_name == f"{base_name}/unit"


def test_log_path_joins_log_dir_and_exp_name():
    exp = DummyExp()
    exp.suffix = "check"
    exp.log_dir = "/tmp/logs"

    exp_name = str(exp.exp_name)
    assert exp.log_path == os.path.join("/tmp/logs", exp_name)


def test_update_from_args_calls_parse_args_and_merge(monkeypatch):
    exp = DummyExpWithMerge()
    parsed_args = {"foo": "bar"}
    parse_called = {"count": 0}

    def fake_parse_args():
        parse_called["count"] += 1
        return parsed_args

    def fake_setup_logger(_log_path):
        return None

    steptron_module = types.ModuleType("steptronoss")
    steptron_utils_module = types.ModuleType("steptronoss.utils")
    arguments_module = types.ModuleType("steptronoss.utils.arguments")
    logger_module = types.ModuleType("steptronoss.utils.logger")
    arguments_module.parse_args = fake_parse_args
    logger_module.setup_logger = fake_setup_logger
    steptron_utils_module.arguments = arguments_module
    steptron_utils_module.logger = logger_module
    steptron_module.utils = steptron_utils_module

    monkeypatch.setitem(sys.modules, "steptronoss", steptron_module)
    monkeypatch.setitem(sys.modules, "steptronoss.utils", steptron_utils_module)
    monkeypatch.setitem(sys.modules, "steptronoss.utils.arguments", arguments_module)
    monkeypatch.setitem(sys.modules, "steptronoss.utils.logger", logger_module)

    exp.update_from_args()

    assert parse_called["count"] == 1
    assert exp.merge_args == parsed_args
    assert exp.merge_kwargs == {"exists_only": True}
