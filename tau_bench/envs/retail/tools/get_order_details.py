# Copyright Sierra

import json
from typing import Any, Dict
from tau_bench.envs.tool import Tool


class GetOrderDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str) -> str:
        orders = data["orders"]
        if order_id in orders:
            # return (orders[order_id])
            return json.dumps(orders[order_id])
        return "Error: order not found"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_order_details",
                "description": "Get the status and details of an order. The output is a dictonary with the fields: 'order_id', 'user_id', 'address', 'items', 'fulfillments', 'status', 'payment_history'. Items is a list of dicts with keys: name, product_id, item_id, price, options. Options is further a dictionary with details. If the order is not found, the output is a string saying that the order isn't found",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                        },
                    },
                    "required": ["order_id"],
                },
            },
        }
