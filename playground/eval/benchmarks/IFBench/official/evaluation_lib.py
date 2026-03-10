# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation helpers used by the OSS IFBench benchmark wrapper."""

import collections
import dataclasses

from . import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, str | int | None]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


@dataclasses.dataclass
class AccuracyReport:
    prompt_level_accuracy: float
    instruction_level_accuracy: float
    tier0_accuracy: dict[str, float]
    tier1_accuracy: dict[str, float]

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "prompt_level_accuracy": self.prompt_level_accuracy,
            "instruction_level_accuracy": self.instruction_level_accuracy,
            "tier0_accuracy": dict(self.tier0_accuracy),
            "tier1_accuracy": dict(self.tier1_accuracy),
        }


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        inp.kwargs[index] = {key: value for key, value in inp.kwargs[index].items() if value is not None}
        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response and response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    if response is None:
        return OutputExample(
            instruction_id_list=inp.instruction_id_list,
            prompt=inp.prompt,
            response="",
            follow_all_instructions=False,
            follow_instruction_list=[False] * len(inp.instruction_id_list),
        )

    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def build_accuracy_report(outputs: list[OutputExample]) -> AccuracyReport:
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total: dict[str, int] = collections.defaultdict(int)
    tier0_correct: dict[str, int] = collections.defaultdict(int)
    tier1_total: dict[str, int] = collections.defaultdict(int)
    tier1_correct: dict[str, int] = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list, strict=True):
            tier0_id = instruction_id.split(":")[0]
            tier0_total[tier0_id] += 1
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[tier0_id] += 1
                tier1_correct[instruction_id] += 1

    return AccuracyReport(
        prompt_level_accuracy=prompt_correct / max(prompt_total, 1),
        instruction_level_accuracy=instruction_correct / max(instruction_total, 1),
        tier0_accuracy={
            instruction_id: tier0_correct[instruction_id] / tier0_total[instruction_id]
            for instruction_id in sorted(tier0_total)
        },
        tier1_accuracy={
            instruction_id: tier1_correct[instruction_id] / tier1_total[instruction_id]
            for instruction_id in sorted(tier1_total)
        },
    )
