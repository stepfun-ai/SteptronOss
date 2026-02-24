from enum import Enum
from typing import Any, Literal, TypedDict


class Role(str, Enum):
    SYSTEM = "system"
    HUMAN = "human"
    USER = "user"
    ASSISTANT = "assistant"
    CONTEXT = "context"  # context
    FUNCTION = "function"  # function definition


class MessageContentType(str, Enum):
    TEXT = "text"
    CALL = "call"
    TOKEN = "token"


VALID_MESSAGE_CONTENT_TYPES = {m.value for m in MessageContentType}


class MessageContentPart(TypedDict):
    type: MessageContentType
    value: str
    loss_mask: Literal[0, 1] | None


class MessageItem(TypedDict):
    role: Role
    content: list[MessageContentPart]
    name: str | None
    loss_mask: Literal[0, 1]
    ground_truth: Any | None


class Dialog(TypedDict):
    conversations: list[MessageItem]


Dataset = list[Dialog]
