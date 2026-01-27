"""Weight-loading util that load a HuggingFace safetensor weight
and map keys according to Indicator.

An indicator can be a string or a list of string like this
```
model.*: transformer.model.*
*wqkv_proj*: CAT(0) *q_proj* *k_proj* *v_proj*"
```
This means remove prefix "transformer." from all keys and merge
q/k/v proj into one tensor.

For identity mapping, use "*: *".

Example:
```
safe_tensors = HFWeight("/data/Qwen3-8B/")
# {"transformer.model.a.weight": Tensor}
state_dict = safe_tensors.export("model.*: transformer.model.*")
# {"model.a.weight": Tensor}
```

"""

import re

import torch
from loguru import logger

from steptronoss.core.parallel_state import PM

__all__ = ["HFWeights"]


def translate(pat):
    # Copied from fnmatch.translate, remove multi-match logic
    STAR = object()
    res = []
    add = res.append
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        i = i + 1
        if c == "*":
            # compress consecutive `*` into one
            if (not res) or res[-1] is not STAR:
                add(STAR)
        elif c == "?":
            add(".")
        elif c == "[":
            j = i
            if j < n and pat[j] == "!":
                j = j + 1
            if j < n and pat[j] == "]":
                j = j + 1
            while j < n and pat[j] != "]":
                j = j + 1
            if j >= n:
                add("\\[")
            else:
                stuff = pat[i:j]
                if "-" not in stuff:
                    stuff = stuff.replace("\\", r"\\")
                else:
                    chunks = []
                    k = i + 2 if pat[i] == "!" else i + 1
                    while True:
                        k = pat.find("-", k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k + 1
                        k = k + 3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += "-"
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks) - 1, 0, -1):
                        if chunks[k - 1][-1] > chunks[k][0]:
                            chunks[k - 1] = chunks[k - 1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = "-".join(s.replace("\\", r"\\").replace("-", r"\-") for s in chunks)
                # Escape set operations (&&, ~~ and ||).
                stuff = re.sub(r"([&~|])", r"\\\1", stuff)
                i = j + 1
                if not stuff:
                    # Empty range: never match.
                    add("(?!)")
                elif stuff == "!":
                    # Negated empty range: match any character.
                    add(".")
                else:
                    if stuff[0] == "!":
                        stuff = "^" + stuff[1:]
                    elif stuff[0] in ("^", "["):
                        stuff = "\\" + stuff
                    add(f"[{stuff}]")
        else:
            add(re.escape(c))
    assert i == n

    # Deal with STARs.
    inp = res
    res = []
    add = res.append
    i, n = 0, len(inp)
    # Fixed pieces at the start?
    while i < n and inp[i] is not STAR:
        add(inp[i])
        i += 1
    while i < n:
        assert inp[i] is STAR
        i += 1
        if i == n:
            add("(.*)")
            break
        assert inp[i] is not STAR
        fixed = []
        while i < n and inp[i] is not STAR:
            fixed.append(inp[i])
            i += 1
        fixed = "".join(fixed)

        add("(.*)")
        add(fixed)

    assert i == n
    res = "".join(res)
    return rf"(?s:{res})\Z"


def apply_match(match: re.Match, string: str):
    if "*" in string:
        assert "$" not in string
        string = "".join([f"${i}" + x for i, x in enumerate(string.split("*"))])
        string = string[2:]
    stars = match.groups()
    for gi, g in enumerate(stars):
        string = string.replace(f"${gi+1}", g)
    return string


def parse_indicator_from_str(indicator: list[str] | str):
    result = {}
    if isinstance(indicator, str):
        indicator = indicator.splitlines()
    for code in indicator:
        if ":" in code:
            name, cmd = code.split(":")
            cmd = cmd.lstrip(" ").split(" ")
            if len(cmd) == 1:  # Is
                result[name] = Is(cmd[0])
            else:
                op = cmd[0]
                if op.startswith("CAT"):
                    dim = re.match("CAT\((\d)\)", op)
                    assert dim, f"Illegal OP: {cmd}"
                    dim = int(dim.group(1))
                    result[name] = Cat(*cmd[1:], dim=dim)
                else:
                    raise RuntimeError(f"Illegal OP: {cmd}")
    return result


class Is:
    def __init__(self, ref_name: str):
        op = None
        if "|" in ref_name:
            ref_name, op = ref_name.split("|")
        self.ref_name = ref_name

        self.op = None
        if op:
            t, d = op.split("@")
            self.op = (t, int(d))


class Cat:
    def __init__(self, *comps, dim=0, pop=True):
        assert comps
        self.dim = dim
        self.comps = comps
        self.pop = pop


class HFWeights(dict):
    def __init__(self, path: str):
        import json
        from os.path import exists, join

        from safetensors.torch import load_file

        index_file = join(path, "model.safetensors.index.json")
        if exists(index_file):
            with open(index_file, "r") as f:
                mapper = json.load(f)["weight_map"]
            weight_files = dict()
            for k, v in mapper.items():
                if v not in weight_files:
                    weight_files[v] = load_file(join(path, v))
                self[k] = weight_files[mapper[k]][k]
        else:
            data = load_file(join(path, "model.safetensors"))
            for k, v in data.items():
                self[k] = v

    def export(self, indicator: list[str] | str | dict[str, Is | Cat]) -> dict[str, torch.Tensor]:
        """Export the state_dict to new naming. An naming indicator is required.

        Args:
            indicator (list[str] | str): Naming indicator,
            e.g.
            ```
            '''
                model.*: transformer.model.*
                *wqkv_proj* CAT(0) *q_proj* *k_proj* *v_proj*"
            '''
            ```
            means remove prefix "transformer." from all keys and merge q/k/v proj into one tensor.

        Returns:
            dict: Processed Dict.
        """
        if not isinstance(indicator, dict):
            indicator = parse_indicator_from_str(indicator)
        patterns = [translate(x.ref_name) for x in indicator.values() if isinstance(x, Is)]
        ops = [x.op for x in indicator.values() if isinstance(x, Is)]
        keys = [k for k, v in indicator.items() if isinstance(v, Is)]
        out = {}

        for k in self:
            dst_name = k
            dst_value: torch.Tensor = self[k]
            for i, p in enumerate(patterns):
                if m := re.match(p, k):
                    dst_name = keys[i]
                    dst_name = apply_match(m, dst_name)

                    if ops[i]:
                        P, D = ops[i]
                        dim = dst_value.shape[D]
                        chunk_dim = dim // PM.size_of(P)
                        all_slice = [slice(None, None)] * len(dst_value.shape)
                        all_slice[D] = slice(
                            chunk_dim * PM.rank_in(P),
                            chunk_dim * PM.rank_in(P) + chunk_dim,
                        )
                        dst_value = dst_value[all_slice]
                    break

            out[dst_name] = dst_value

        merges = {k: v for k, v in indicator.items() if isinstance(v, Cat)}

        now_keys = list(out)
        for k in now_keys:
            for dst, mg in merges.items():
                p = translate(mg.comps[0])
                if m := re.match(p, k):
                    data = []
                    for c in mg.comps:
                        cc = apply_match(m, c)
                        if mg.pop:
                            data.append(out.pop(cc))
                        else:
                            data.append(out.get(cc))
                    d = apply_match(m, dst)
                    out[d] = torch.cat(data, mg.dim)
                    break

        return out

    def write_safetensors_to(self, path: str, shard_max_size=10e9):
        import json
        from os import makedirs
        from os.path import join

        from safetensors.torch import save_file

        from steptronoss.utils.general import convert_num

        makedirs(path, exist_ok=True)

        total_size, weight_map = 0, {}
        buffer_size, buffer, shard_idx = 0, {}, 1

        file_name = f"model-{shard_idx:05d}.safetensors"

        for k, v in self.items():
            v_size = v.numel() * v.dtype.itemsize
            if buffer_size and (buffer_size + v_size > shard_max_size):
                full_path = join(path, file_name)
                save_file(buffer, full_path)
                logger.info(f"{convert_num(buffer_size, G=True)} data dumped to {full_path}")
                buffer.clear()
                buffer_size = 0
                shard_idx += 1
                file_name = f"model-{shard_idx:05d}.safetensors"

            weight_map[k] = file_name
            buffer[k] = v
            buffer_size += v_size
            total_size += v_size

        if buffer_size:
            full_path = join(path, file_name)
            save_file(buffer, full_path)
            logger.info(f"{convert_num(buffer_size, G=True)} data dumped to {full_path}")
            buffer.clear()
            buffer_size = 0

        with open(join(path, "model.safetensors.index.json"), "w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                },
                f,
            )
