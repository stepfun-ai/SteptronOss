IM_END_TOKEN = r"<|im_end|>"
THINK_END_WITH_NEWLINE = "</think>\n"
THINK_BEGIN_TOKEN = r"<think>"
THINK_END_TOKEN = r"</think>"


def parse_chat_reasoning_content(response: str) -> str:
    return ""


def parse_chat_solution(response: str) -> str:
    parts = response.split(IM_END_TOKEN)
    if len(parts) > 1:
        return parts[0]
    return ""


def parse_reason_reasoning_content(response: str) -> str:
    end_splits = response.split(THINK_END_TOKEN)
    if len(end_splits) >= 2:
        content = "".join(end_splits[:-1])
        begin_splits = content.split(THINK_BEGIN_TOKEN)
        if len(begin_splits) >= 2:
            content = "".join(begin_splits[1:])
        else:
            content = "".join(begin_splits)
    else:
        content = response
    return content.strip("\n")


def parse_reason_solution(response: str) -> str:
    boa_splits = response.split(THINK_END_WITH_NEWLINE)
    if len(boa_splits) > 1:
        boa_split = boa_splits[-1]
        eoa_splits = boa_split.split(IM_END_TOKEN)
        if len(eoa_splits) > 1:
            return eoa_splits[0]
    return ""
