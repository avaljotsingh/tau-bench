# Copyright Sierra

from typing import Any, Dict
from tau_bench.envs.tool import Tool


class GetInputFromUser(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], thought: str) -> str:
        # This method does not change the state of the data; it simply returns an empty string.
        return ""

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_input_from_user",
                "description": (
                    "Use the tool to get input from user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "A thought to think about.",
                        },
                    },
                    "required": ["thought"],
                },
            },
        }
