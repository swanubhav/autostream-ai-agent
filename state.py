from typing import TypedDict, Optional

class AgentState(TypedDict):
    messages: list
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    response: Optional[str]