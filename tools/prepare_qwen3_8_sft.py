"""Prepare assistant-only text JSONL before starting distributed training."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from steptronoss.data.qwen3_8_sft import encode_chat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "xhigh"], default="medium")
    args = parser.parse_args()
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if not tokenizer.is_fast:
        parser.error("A fast tokenizer is required for exact assistant offset masking")
    source, output = Path(args.input), Path(args.output)
    if source.resolve() == output.resolve():
        parser.error("Input and output must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    count, truncated = 0, 0
    try:
        with source.open() as fin, temporary.open("x") as fout:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    sample = json.loads(line)
                    record = encode_chat(
                        tokenizer,
                        sample,
                        args.max_length,
                        {
                            "enable_thinking": args.enable_thinking,
                            "reasoning_effort": args.reasoning_effort,
                        },
                    )
                except Exception as exc:
                    raise ValueError(f"Invalid training record at line {line_number}: {exc}") from exc
                truncated += len(record["input_ids"]) == args.max_length
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if not count:
            raise ValueError("Input contains no training records")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "format": "steptron-qwen38-text-v1",
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "max_length": args.max_length,
        "records": count,
        "at_length_limit": truncated,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "output": str(output),
        "model": args.model,
        "revision": args.revision,
    }
    Path(str(output) + ".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
