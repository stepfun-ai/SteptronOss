# Tests

## Quick start

```bash
# Activate venv if you have one
source ./.venv/bin/activate

# Run CPU-marked tests only (avoids skip noise)
pytest -m cpu
```

## GPU-only tests

Some tests require CUDA and `flash_attn`. If you see skips like
"could not import 'flash_attn'", install it in your GPU environment first.

```bash
# Run all GPU-marked tests (no skip noise)
pytest -m gpu
```

## Multi-process (node2 / torchrun) tests

These tests require `WORLD_SIZE=2`. Run them with `torchrun`:

```bash
torchrun --nproc-per-node=2 -m pytest -m node2
```

## Notes

- Some tests are expected to skip on CPU-only machines.
- If you see warnings about `xdist_group`, they can be ignored unless you are
  running with `pytest-xdist`.
