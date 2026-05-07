import torch

from steptronoss.checkpointing.reshape_ops import Identity, OnlineReshaper, Script
from steptronoss.model.deepseek_v4 import DeepseekV4FlashExperts, DeepseekV4FlashFP8Dequant


def test_online_reshaper_exact_key_source():
    weights = {
        "a.weight": torch.ones(1),
        "b.weight": torch.full((1,), 2.0),
    }

    out = OnlineReshaper([Script(src="a.weight", op=Identity(), dst="a.weight")]).forward(weights)

    assert list(out) == ["a.weight"]
    assert out["a.weight"].item() == 1.0


def test_deepseek_v4_flash_fp8_block_dequant():
    raw = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.full((1, 1), 0.5, dtype=torch.float32)

    out = DeepseekV4FlashFP8Dequant().forward({"x.weight": raw, "x.scale": scale})

    assert out["x.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(out["x.weight"].float(), raw.float() * 0.5)


def test_deepseek_v4_flash_expert_release_keys_stack_by_expert_id():
    piece = {
        "layers.0.ffn.experts.3.w1.weight": torch.ones(2, 3),
        "layers.0.ffn.experts.3.w3.weight": torch.full((2, 3), 3.0),
        "layers.0.ffn.experts.5.w1.weight": torch.full((2, 3), 5.0),
        "layers.0.ffn.experts.5.w3.weight": torch.full((2, 3), 7.0),
        "layers.0.ffn.experts.3.w2.weight": torch.full((3, 2), 11.0),
        "layers.0.ffn.experts.5.w2.weight": torch.full((3, 2), 13.0),
    }

    gate_up = DeepseekV4FlashExperts("mlp.experts.gate_up_proj", gate_up=True).forward(piece)
    down = DeepseekV4FlashExperts("mlp.experts.down_proj", gate_up=False).forward(piece)

    gate_up = gate_up["mlp.experts.gate_up_proj"]
    down = down["mlp.experts.down_proj"]
    assert gate_up.shape == (2, 4, 3)
    assert down.shape == (2, 3, 2)
    assert gate_up[0, :2].eq(1).all()
    assert gate_up[0, 2:].eq(3).all()
    assert gate_up[1, :2].eq(5).all()
    assert gate_up[1, 2:].eq(7).all()
    assert down[0].eq(11).all()
    assert down[1].eq(13).all()
