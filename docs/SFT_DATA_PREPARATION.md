# SFT 数据准备与编译

本文说明如何组织 SFT 数据并用于训练，参考实现见：
- `steptronoss/data/recipe.py`
- `playground/tools/compile_recipe.py`

## 流程概览

1) 准备 JSON 数据文件（参考格式）。
2) 编写 `CompliableDatasetsConfig`，描述数据域与采样。
3) （可选）编译 datasets，生成 compiled 数据路径（仅用于加速读取）。
4) （可选）改用 `CompiledDatasetsConfig`。
5) 编写 `SFTDataConfig` 并接入实验。

## 1) JSON 数据格式

`StepChatJsonDataset` 读取的格式是 **JSON 数组**，每个元素是一条对话样本。

最小结构示意（字段名仅作结构参考，不包含具体内容）：

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": null
}
```

更详细的 item 结构示意（字段名仅作结构参考，不包含具体内容）：

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

要点：
- 根结构是数组；每个元素包含 `conversations`。
- `conversations` 是消息列表；每条消息含 `role`，可选 `name`/`loss_mask`/`ground_truth`。
- `content` 既可为字符串，也可为 list（多段内容，带 `type/value`）。
- `loss_mask` 在 `content` 颗粒度也可出现（覆盖对应片段的训练权重）。
- `tool_schemas`/`tools` 可选；用于工具调用相关样本。
- `images` 可省略或为 `null`。

如需对照格式，请在内部数据目录中选择任意一份现有数据作为参考。

## 2) 编写 CompliableDatasetsConfig

创建一个可编译的数据配置类（示例路径：`playground/data/sft/my_recipe.py`）：

```python
from playground.tools.compile_recipe import CompliableDatasetsConfig
from steptronoss.data.recipe import DataRecipe, DataSourceFile


class MyDatasetsConfig(CompliableDatasetsConfig):
    def get_template(self):
        # 可选：返回一个 dialog template（无特殊需求可返回 None）
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

说明：
- `domains`：按域组织文件列表（每个文件是 `DataSourceFile`）。
- `subsample_rate` 范围为 (0, 1]，用于下采样。
- `epochs`：采样策略（按域设置 epoch 权重），和 `Megatron-LM`参数语义保持一致。

  > 例如 `general` 有 1000 条数据，`subsample_rate=0.3,epoch=2`；`math` 有 500 条数据，DatasetsConfig 共 `1000*0.3+500` 条，随后在 dataloader 进行加权采样，最终生成 `1000*0.3*2+500*1`条数据。

## 3) （可选）编译 datasets

编译会把原始 JSON 转成 compiled 格式（更快读取），并在日志中输出可复制的
`CompiledDatasetsConfig` 代码片段。**注意：编译是可选步骤，仅用于加速读取。**

示例（可在任意脚本或交互中运行）：

```python
from playground.data.sft.my_recipe import MyDatasetsConfig

MyDatasetsConfig().compile("/oss/data/my_sft_compiled")
```

编译完成后日志会打印类似：
```
class DatasetsConfig(CompiledDatasetsConfig):
    compiled_recipe = CompiledDataRecipe(
        domains={...},
        epochs={...},
    )
```

## 4) 改用 CompiledDatasetsConfig

将上一步日志输出的代码粘贴为你的编译后数据配置：

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

## 5) 编写 SFTDataConfig 并接入实验

在你的实验配置中引用 compiled 数据集，并构建 dataloader：

```python
from steptronoss.exp.sft import SFTDataConfig
from playground.data.sft.my_compiled_recipe import MyCompiledDatasetsConfig


class MySFTDataConfig(SFTDataConfig):
    dataset_cfg = MyCompiledDatasetsConfig

    def build_dataloader(self, dp_rank=0, dp_size=1):
        # 参考 playground/data/sft/step_recipe0226.py 的实现
        raise NotImplementedError
```

最后在实验配置（`Exp`）里将 `data_cfg` 指向你的 `MySFTDataConfig` 即可开始训练。

## Data Flow 提示

- `build_dataloader()` 返回 **iterable[Any]**。
- `preprocess()` 需要把这个 `Any` 转换成 `model.forward()` 需要的参数字典。
- 具体字段取决于你的模型与打包策略（例如 `input_ids` / `labels` / `loss_masks` 等）。
