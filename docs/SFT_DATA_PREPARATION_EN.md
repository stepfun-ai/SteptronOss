# SFT Data Preparation and Compilation

This document explains how to organize SFT data and use it for training. Reference implementations:
- `steptronoss/data/recipe.py`
- `playground/tools/compile_recipe.py`

## Workflow Overview

1) Prepare JSON data files (format reference).
2) Write `CompliableDatasetsConfig` to describe domains and sampling.
3) (Optional) compile datasets to generate compiled paths (for faster loading only).
4) Switch to `CompiledDatasetsConfig`.
5) Write `SFTDataConfig` and wire it into your experiment.

## 1) JSON Data Format

`StepChatJsonDataset` expects a **JSON array** where each element is a dialog sample.

Minimal structure (field names are for schema reference only; no real content):

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": null
}
```

More detailed item structure (field names are for schema reference only; no real content):

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "...",
      "name": "...",
      "loss_mask": 1,
      "ground_truth": null
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "value": "...", "loss_mask": 0},
        {"type": "text", "value": "..."}
      ]
    },
    {
      "role": "assistant",
      "content": "...",
      "tool_schemas": [
        {"name": "...", "description": "...", "parameters": {"type": "object", "properties": {}}}
      ]
    }
  ],
  "images": null
}
```

Notes:
- The root is an array; each element contains `conversations`.
- `conversations` is a list of messages; each message has `role` and optional `name`/`loss_mask`/`ground_truth`.
- `content` can be a string or a list of parts (with `type/value`).
- `loss_mask` can also appear at the content-part level (overrides training weight for that part).
- `tool_schemas`/`tools` are optional for tool-call samples.
- `images` can be omitted or `null`.

If you need a format reference, use any internal dataset as a template.

## 2) Write CompliableDatasetsConfig

Create a compilable dataset config (example path: `playground/data/sft/my_recipe.py`):

```python
from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.data.recipe import DataRecipe, DataSourceFile


class MyDatasetsConfig(CompliableDatasetsConfig):
    def get_template(self):
        # Optional: return a dialog template (return None if not needed)
        return None

    def get_recipe(self) -> DataRecipe:
        return DataRecipe(
            domains={
                "general": [
                    DataSourceFile(
                        path="/mnt/path/to/general.json",
                        subsample_rate=1.0,
                    ),
                ],
                "math": [
                    DataSourceFile(path="/mnt/path/to/math.json"),
                ],
            },
            epochs={
                "general": 2.0,
                "math": 1.0,
            },
        )
```

Notes:
- `domains`: per-domain file lists (each file is a `DataSourceFile`).
- `subsample_rate` must be in (0, 1] for downsampling.
- `epochs`: sampling plan (epoch weight per domain), consistent with Megatron-LM parameter semantics.

  > For example, `general` has 1000 samples with subsample_rate=0.3, epochs=2; `math` has 500 samples with default settings. The `DatasetsConfig` contains `1000*0.3+500` entries, which are then weighted-sampled in the dataloader, ultimately producing `1000*0.3*2+500*1` samples.

## 3) (Optional) Compile Datasets

Compilation converts raw JSON into compiled format for faster loading and prints a
copy-paste `CompiledDatasetsConfig` snippet in the logs. **Compilation is optional and only for speed.**

Example (run in any script or REPL):

```python
from playground.data.sft.my_recipe import MyDatasetsConfig

MyDatasetsConfig().compile("/oss/data/my_sft_compiled")
```

The logs will print something like:
```
class DatasetsConfig(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={...},
        epochs={...},
    )
```

## 4) Switch to CompiledDatasetsConfig

Paste the generated code into your compiled datasets config:

```python
from playground.tools.compile_recipe import CompiledDataRecipe, CompiledDatasetsConfig


class MyCompiledDatasetsConfig(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={
            "general": "/oss/data/my_sft_compiled/general",
            "math": "/oss/data/my_sft_compiled/math",
        },
        epochs={
            "general": 2.0,
            "math": 1.0,
        },
    )
```

## 5) Write SFTDataConfig and Wire Into Experiment

Reference the compiled datasets in your experiment config and build the dataloader:

```python
from steptronoss.exp.sft import SFTDataConfig
from playground.data.sft.my_compiled_recipe import MyCompiledDatasetsConfig


class MySFTDataConfig(SFTDataConfig):
    dataset_cfg = MyCompiledDatasetsConfig

    def build_dataloader(self, dp_rank=0, dp_size=1):
        # See playground/data/sft/step_recipe0226.py for a concrete implementation
        raise NotImplementedError
```

Finally, set `data_cfg` to your `MySFTDataConfig` in the experiment (`Exp`).

## Data Flow Note

- `build_dataloader()` returns **iterable[Any]**.
- `preprocess()` must convert that `Any` into the argument dict required by `model.forward()`.
- The exact fields depend on your model and packing strategy (e.g., `input_ids` / `labels` / `loss_masks`).
