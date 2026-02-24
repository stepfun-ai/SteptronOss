import os

from tools import build_scripts


def test_build_scripts_generates_replica_scripts(tmp_path):
    exp_path = "tests/fixtures/build_scripts_exp.py"
    result = build_scripts.build_scripts(
        exp_path=exp_path,
        output_root=str(tmp_path),
        extra_args=["suffix=test"],
        exp_id="abc12",
    )

    exp_root = tmp_path / "build_scripts_exp" / "test" / "abc12"
    assert exp_root.is_dir()
    assert result.output_dir == str(exp_root)
    assert len(result.scripts) == 7

    train_script = exp_root / "gpu" / "0.sh"
    assert train_script.is_file()
    assert os.access(train_script, os.X_OK)
    content = train_script.read_text(encoding="utf-8")
    assert "export EXP_ID=abc12/train" in content
    assert "export FOO=BAR" in content
    assert "export TRAIN_ONLY=1" in content
    assert "export NNODES=2" in content
    assert "tools/smartrun tests/fixtures/build_scripts_exp.py suffix=test" in content

    infer_scripts = sorted((exp_root / "gpu").glob("*.sh"))
    assert len(infer_scripts) == 6
    infer_content = (exp_root / "gpu" / "2.sh").read_text(encoding="utf-8")
    assert "export EXP_ID=abc12/infer" in infer_content

    control_script = exp_root / "cpu" / "0.sh"
    assert control_script.is_file()
