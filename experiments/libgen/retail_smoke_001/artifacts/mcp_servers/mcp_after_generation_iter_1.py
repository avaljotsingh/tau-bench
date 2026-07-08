from mcp.server.fastmcp import FastMCP
from typing import Any, Dict, List
import json
from tau_bench.envs.my_data import global_data
import builtins
mcp = FastMCP('MCP server for retail env')

import logging

logging.basicConfig(filename='mcp_debug.log', level=logging.DEBUG)


@mcp.tool()
def calculate(expression: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate the result of a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
                    },
                },
                "required": ["expression"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    if not all(char in "0123456789+-*/(). " for char in expression):
        return "Error: invalid characters in expression"
    try:
        # Evaluate the mathematical expression safely
        # return (round(float(eval(expression, {"__builtins__": None}, {})), 2))
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def get_user_details(user_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Get the details of a user, including their orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user id, such as 'sara_doe_496'.",
                    },
                },
                "required": ["user_id"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    users = data["users"]
    if user_id in users:
        return json.dumps(users[user_id])
    return "Error: user not found"

@mcp.tool()
def get_order_details(order_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get the status and details of an order.",
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
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    orders = data["orders"]
    if order_id in orders:
        return json.dumps(orders[order_id])
    return "Error: order not found"

@mcp.tool()
def get_product_details(product_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get the inventory details of a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                    },
                },
                "required": ["product_id"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    products = data["products"]
    if product_id in products:
        return json.dumps(products[product_id])
    return "Error: product not found"

@mcp.tool()
def find_user_id_by_email(email: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_email",
            "description": "Find user id by email. If the user is not found, the function will return an error message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email of the user, such as 'something@example.com'.",
                    },
                },
                "required": ["email"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    users = data["users"]
    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            return user_id
    return "Error: user not found"

@mcp.tool()
def find_user_id_by_name_zip(first_name: str, last_name: str, zip: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "find_user_id_by_name_zip",
            "description": "Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "first_name": {
                        "type": "string",
                        "description": "The first name of the customer, such as 'John'.",
                    },
                    "last_name": {
                        "type": "string",
                        "description": "The last name of the customer, such as 'Doe'.",
                    },
                    "zip": {
                        "type": "string",
                        "description": "The zip code of the customer, such as '12345'.",
                    },
                },
                "required": ["first_name", "last_name", "zip"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    users = data["users"]
    for user_id, profile in users.items():
        if (
            profile["name"]["first_name"].lower() == first_name.lower()
            and profile["name"]["last_name"].lower() == last_name.lower()
            and profile["address"]["zip"] == zip
        ):
            return user_id
    return "Error: user not found"

@mcp.tool()
def list_all_product_types() -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "list_all_product_types",
            "description": "List the name and product id of all product types. Each product type has a variety of different items with unique item ids and options. There are only 50 product types in the store.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    products = data["products"]
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    return json.dumps(product_dict)

@mcp.tool()
def cancel_pending_order(order_id: str, reason: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "cancel_pending_order",
            "description": "Cancel a pending order. If the order is already processed or delivered, it cannot be cancelled. The agent needs to explain the cancellation detail and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, the order status will be changed to 'cancelled' and the payment will be refunded. The refund will be added to the user's gift card balance immediately if the payment was made using a gift card, otherwise the refund would take 5-7 business days to process. The function returns the order details after the cancellation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["no longer needed", "ordered by mistake"],
                        "description": "The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.",
                    },
                },
                "required": ["order_id", "reason"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    # check order exists and is pending
    orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be cancelled"

    # check reason
    if reason not in ["no longer needed", "ordered by mistake"]:
        return "Error: invalid reason"

    # handle refund
    refunds = []
    for payment in order["payment_history"]:
        payment_id = payment["payment_method_id"]
        refund = {
            "transaction_type": "refund",
            "amount": payment["amount"],
            "payment_method_id": payment_id,
        }
        refunds.append(refund)
        if "gift_card" in payment_id:  # refund to gift card immediately
            payment_method = data["users"][order["user_id"]]["payment_methods"][
                payment_id
            ]
            payment_method["balance"] += payment["amount"]
            payment_method["balance"] = round(payment_method["balance"], 2)

    # update order status
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["payment_history"].extend(refunds)

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    return json.dumps(order)

@mcp.tool()
def get_input_from_user(thought: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "get_input_from_user",
            "description": "Use the tool to get input from user.",
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
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    # This method does not change the state of the data; it simply returns an empty string.
    return ""

@mcp.tool()
def think(thought: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
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
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    # This method does not change the state of the data; it simply returns an empty string.
    return ""

@mcp.tool()
def transfer_to_human_agents(summary: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A summary of the user's issue.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    # This method simulates the transfer to a human agent.
    return "Transfer successful"

@mcp.tool()
def modify_pending_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "modify_pending_order_items",
            "description": "Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "item_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.",
                    },
                    "new_item_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.",
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                    },
                },
                "required": [
                    "order_id",
                    "item_ids",
                    "new_item_ids",
                    "payment_method_id",
                ],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    products, orders, users = data["products"], data["orders"], data["users"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the items to be modified exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # Check new items exist, match old items, and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    # Check if the payment method exists
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # If the new item is more expensive, check if the gift card has enough balance
    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the new item"

    # Handle the payment or refund
    order["payment_history"].append(
        {
            "transaction_type": "payment" if diff_price > 0 else "refund",
            "amount": abs(diff_price),
            "payment_method_id": payment_method_id,
        }
    )
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= diff_price
        payment_method["balance"] = round(payment_method["balance"], 2)

    # Modify the order
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        item["item_id"] = new_item_id
        item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
        item["options"] = products[item["product_id"]]["variants"][new_item_id]["options"]
    order["status"] = "pending (item modified)"

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(order)

@mcp.tool()
def exchange_delivered_order_items(order_id: str, item_ids: List[str], new_item_ids: List[str], payment_method_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "exchange_delivered_order_items",
            "description": "Exchange items in a delivered order to new items of the same product type. For a delivered order, return or exchange can be only done once by the agent. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "item_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.",
                    },
                    "new_item_ids": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The item ids to be exchanged for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.",
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                    },
                },
                "required": [
                    "order_id",
                    "item_ids",
                    "new_item_ids",
                    "payment_method_id",
                ],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    products, orders, users = data["products"], data["orders"], data["users"]

    # check order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be exchanged"

    # check the items to be exchanged exist
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return f"Error: {item_id} not found"

    # check new items exist and match old items and are available
    if len(item_ids) != len(new_item_ids):
        return "Error: the number of items to be exchanged should match"

    diff_price = 0
    for item_id, new_item_id in zip(item_ids, new_item_ids):
        item = [item for item in order["items"] if item["item_id"] == item_id][0]
        product_id = item["product_id"]
        if not (
            new_item_id in products[product_id]["variants"]
            and products[product_id]["variants"][new_item_id]["available"]
        ):
            return f"Error: new item {new_item_id} not found or available"

        old_price = item["price"]
        new_price = products[product_id]["variants"][new_item_id]["price"]
        diff_price += new_price - old_price

    diff_price = round(diff_price, 2)

    # check payment method exists and can cover the price difference if gift card
    if payment_method_id not in users[order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    payment_method = users[order["user_id"]]["payment_methods"][payment_method_id]
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < diff_price
    ):
        return "Error: insufficient gift card balance to pay for the price difference"

    # modify the order
    order["status"] = "exchange requested"
    order["exchange_items"] = sorted(item_ids)
    order["exchange_new_items"] = sorted(new_item_ids)
    order["exchange_payment_method_id"] = payment_method_id
    order["exchange_price_difference"] = diff_price

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(order)

@mcp.tool()
def return_delivered_order_items(order_id: str, item_ids: List[str], payment_method_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "return_delivered_order_items",
            "description": "Return some items of a delivered order. The order status will be changed to 'return requested'. The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. The user will receive follow-up email for how and where to return the item.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.",
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                    },
                },
                "required": ["order_id", "item_ids", "payment_method_id"],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    orders = data["orders"]

    # Check if the order exists and is delivered
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "delivered":
        return "Error: non-delivered order cannot be returned"

    # Check if the payment method exists and is either the original payment method or a gift card
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"
    if (
        "gift_card" not in payment_method_id
        and payment_method_id != order["payment_history"][0]["payment_method_id"]
    ):
        return "Error: payment method should be either the original payment method or a gift card"

    # Check if the items to be returned exist (there could be duplicate items in either list)
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return "Error: some item not found"

    # Update the order status
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(order)

@mcp.tool()
def modify_pending_order_address(order_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "modify_pending_order_address",
            "description": "Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "address1": {
                        "type": "string",
                        "description": "The first line of the address, such as '123 Main St'.",
                    },
                    "address2": {
                        "type": "string",
                        "description": "The second line of the address, such as 'Apt 1' or ''.",
                    },
                    "city": {
                        "type": "string",
                        "description": "The city, such as 'San Francisco'.",
                    },
                    "state": {
                        "type": "string",
                        "description": "The state, such as 'CA'.",
                    },
                    "country": {
                        "type": "string",
                        "description": "The country, such as 'USA'.",
                    },
                    "zip": {
                        "type": "string",
                        "description": "The zip code, such as '12345'.",
                    },
                },
                "required": [
                    "order_id",
                    "address1",
                    "address2",
                    "city",
                    "state",
                    "country",
                    "zip",
                ],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    # Check if the order exists and is pending
    orders = data["orders"]
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Modify the address
    order["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(order)

@mcp.tool()
def modify_pending_order_payment(order_id: str, payment_method_id: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "modify_pending_order_payment",
            "description": "Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.",
                    },
                },
                "required": [
                    "order_id",
                    "payment_method_id",
                ],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    orders = data["orders"]

    # Check if the order exists and is pending
    if order_id not in orders:
        return "Error: order not found"
    order = orders[order_id]
    if order["status"] != "pending":
        return "Error: non-pending order cannot be modified"

    # Check if the payment method exists
    if payment_method_id not in data["users"][order["user_id"]]["payment_methods"]:
        return "Error: payment method not found"

    # Check that the payment history should only have one payment
    if (
        len(order["payment_history"]) > 1
        or order["payment_history"][0]["transaction_type"] != "payment"
    ):
        return "Error: there should be exactly one payment for a pending order"

    # Check that the payment method is different
    if order["payment_history"][0]["payment_method_id"] == payment_method_id:
        return "Error: the new payment method should be different from the current one"

    amount = order["payment_history"][0]["amount"]
    payment_method = data["users"][order["user_id"]]["payment_methods"][
        payment_method_id
    ]

    # Check if the new payment method has enough balance if it is a gift card
    if (
        payment_method["source"] == "gift_card"
        and payment_method["balance"] < amount
    ):
        return "Error: insufficient gift card balance to pay for the order"

    # Modify the payment method
    order["payment_history"].extend(
        [
            {
                "transaction_type": "payment",
                "amount": amount,
                "payment_method_id": payment_method_id,
            },
            {
                "transaction_type": "refund",
                "amount": amount,
                "payment_method_id": order["payment_history"][0]["payment_method_id"],
            },
        ]
    )

    # If payment is made by gift card, update the balance
    if payment_method["source"] == "gift_card":
        payment_method["balance"] -= amount
        payment_method["balance"] = round(payment_method["balance"], 2)

    # If refund is made to a gift card, update the balance
    if "gift_card" in order["payment_history"][0]["payment_method_id"]:
        old_payment_method = data["users"][order["user_id"]]["payment_methods"][
            order["payment_history"][0]["payment_method_id"]
        ]
        old_payment_method["balance"] += amount
        old_payment_method["balance"] = round(old_payment_method["balance"], 2)

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(order)

@mcp.tool()
def modify_user_address(user_id: str, address1: str, address2: str, city: str, state: str, country: str, zip: str) -> str:
    """
    {
        "type": "function",
        "function": {
            "name": "modify_user_address",
            "description": "Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user id, such as 'sara_doe_496'.",
                    },
                    "address1": {
                        "type": "string",
                        "description": "The first line of the address, such as '123 Main St'.",
                    },
                    "address2": {
                        "type": "string",
                        "description": "The second line of the address, such as 'Apt 1' or ''.",
                    },
                    "city": {
                        "type": "string",
                        "description": "The city, such as 'San Francisco'.",
                    },
                    "state": {
                        "type": "string",
                        "description": "The state, such as 'CA'.",
                    },
                    "country": {
                        "type": "string",
                        "description": "The country, such as 'USA'.",
                    },
                    "zip": {
                        "type": "string",
                        "description": "The zip code, such as '12345'.",
                    },
                },
                "required": [
                    "user_id",
                    "address1",
                    "address2",
                    "city",
                    "state",
                    "country",
                    "zip",
                ],
            },
        },
    }
    """
    with open("data.json", "r") as f:
        loaded_data = json.load(f)
    data = loaded_data
    users = data["users"]
    if user_id not in users:
        return "Error: user not found"
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip,
    }
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    return json.dumps(user)
@mcp.tool()
def list_user_payment_methods(user_id):
    """
{
  "type": "function",
  "function": {
    "name": "list_user_payment_methods",
    "description": "Fetches a user's details and compiles a list of their saved payment methods, including type (gift card, credit card, PayPal), masked identifiers, gift card balance (if available), and the payment_method_id values for use in order modifications, returns, and exchanges. Returns an empty list if none are saved.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string",
          "description": "The user id to look up, such as 'sara_doe_496'."
        }
      },
      "required": [
        "user_id"
      ]
    }
  }
}
    """
    import re

    # Validate input
    if user_id is None or not isinstance(user_id, str) or not user_id.strip():
        return {"error": "user_id is required and must be a non-empty string"}

    # Helper functions for masking and parsing

    def _mask_email(email):
        try:
            local, domain = email.split('@', 1)
        except Exception:
            # Fallback: mask last 4 characters if not a normal email
            s = str(email)
            if len(s) <= 4:
                return '*' * len(s)
            return ('*' * (len(s) - 4)) + s[-4:]
        if not local:
            return '*' + '@' + domain
        return local[0] + '***@' + domain


    def _mask_last4(number_like):
        s = str(number_like or '')
        digits = re.sub(r"\D", "", s)
        if len(digits) >= 4:
            return '**** **** **** ' + digits[-4:]
        # If no digits, fallback to generic mask
        if s:
            # Use last 4 visible chars
            tail = s[-4:]
            return '****' + tail
        return '****'


    def _parse_amount(val):
        # Attempt to parse a numeric amount from various shapes
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, dict):
            # Common fields: amount, value, remaining
            for k in ("amount", "value", "remaining"):
                if k in val:
                    return _parse_amount(val[k])
            # If dict has currency+amount-like
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    cleaned = re.sub(r"[^0-9.\-]", "", v)
                    try:
                        return float(cleaned)
                    except Exception:
                        continue
            return None
        # String case
        s = str(val)
        cleaned = re.sub(r"[^0-9.\-]", "", s)
        try:
            return float(cleaned)
        except Exception:
            return s  # return as string if cannot parse


    def _detect_type(pm, forced_type=None):
        if forced_type:
            return forced_type
        candidates = []
        for k in ("type", "method_type", "method", "provider", "brand", "network", "card_type"):
            v = pm.get(k)
            if isinstance(v, str) and v:
                candidates.append(v.lower())
        joined = " ".join(candidates)
        if any(w in joined for w in ["gift", "giftcard", "gc-"]):
            return "gift card"
        if any(w in joined for w in ["paypal", "pay pal"]):
            return "paypal"
        if any(w in joined for w in ["credit", "card", "visa", "mastercard", "amex", "discover"]):
            return "credit card"
        # Heuristic by fields
        if any(k in pm for k in ["last4", "card_number", "number", "exp_month", "exp_year"]):
            return "credit card"
        if any(k in pm for k in ["email", "paypal_email", "payer_id"]):
            return "paypal"
        if any(k in pm for k in ["gift_code", "code", "gift_card_id", "balance"]):
            return "gift card"
        return "unknown"


    def _extract_id(pm):
        for k in ("payment_method_id", "paymentMethodId", "id", "paymentId", "payment_method", "method_id"):
            v = pm.get(k)
            if v:
                return str(v)
        return None


    def _masked_identifier_for(pm, method_type):
        # Prefer provided masked value
        for k in ("masked", "masked_identifier", "display", "label"):
            v = pm.get(k)
            if isinstance(v, str) and v.strip():
                return v
        if method_type == "credit card":
            brand = pm.get("brand") or pm.get("network") or pm.get("card_type") or "Card"
            last4 = pm.get("last4")
            if not last4:
                last4 = pm.get("card_number") or pm.get("number")
            masked = _mask_last4(last4)
            # Compose as "Brand **** **** **** 1234" if possible
            if isinstance(brand, str) and brand:
                return f"{brand} {masked}"
            return masked
        if method_type == "paypal":
            email = pm.get("email") or pm.get("paypal_email") or pm.get("account")
            if email:
                return _mask_email(str(email))
            payer = pm.get("payer_id") or pm.get("id")
            if payer:
                return _mask_last4(payer)
            return "PayPal"
        if method_type == "gift card":
            code = pm.get("code") or pm.get("gift_code") or pm.get("number") or pm.get("gift_card_id")
            if code:
                tail = str(code)[-4:]
                return f"Gift Card ****{tail}"
            return "Gift Card"
        # Fallback for unknown
        any_id = _extract_id(pm)
        if any_id:
            return _mask_last4(any_id)
        return "Payment Method"


    def _extract_balance(pm):
        # Extract and parse balance for gift cards
        for k in ("balance", "remaining", "available_balance"):
            if k in pm:
                return _parse_amount(pm[k])
        # Sometimes nested
        bal = pm.get("wallet", {}).get("balance") if isinstance(pm.get("wallet"), dict) else None
        if bal is not None:
            return _parse_amount(bal)
        return None


    def _process_pm(pm, forced_type=None):
        if not isinstance(pm, dict):
            return None
        method_type = _detect_type(pm, forced_type=forced_type)
        pm_id = _extract_id(pm)
        masked_identifier = _masked_identifier_for(pm, method_type)
        result = {
            "payment_method_id": pm_id or "",
            "method_type": method_type,
            "masked_identifier": masked_identifier,
        }
        if method_type == "gift card":
            bal = _extract_balance(pm)
            if bal is not None:
                result["balance"] = bal
        return result

    # Fetch user details
    try:
        data = get_user_details(user_id=user_id)
    except Exception as e:
        return {"error": f"Failed to fetch user details: {e}"}

    if not isinstance(data, dict):
        return {"error": "Unexpected response from get_user_details"}

    if "error" in data and data["error"]:
        # Pass through underlying error if present
        return {"error": str(data.get("error"))}

    # Collect containers to search for payment methods
    containers = []
    if isinstance(data, dict):
        containers.append(data)
        for k in ("user", "profile", "account", "wallet"):
            v = data.get(k)
            if isinstance(v, dict):
                containers.append(v)
            elif isinstance(v, list):
                # Sometimes a list of wallets etc.
                for item in v:
                    if isinstance(item, dict):
                        containers.append(item)

    # Gather raw payment method entries
    raw_entries = []
    # Generic lists
    generic_keys = ("payment_methods", "paymentMethods", "methods")
    for c in containers:
        for k in generic_keys:
            v = c.get(k)
            if isinstance(v, list):
                raw_entries.extend([(pm, None) for pm in v])

    # Group-specific lists with enforced types
    group_maps = [
        (("gift_cards", "giftCards", "giftcards", "gift_card_accounts"), "gift card"),
        (("credit_cards", "creditCards", "cards", "card_accounts"), "credit card"),
        (("paypal_accounts", "paypal", "paypals"), "paypal"),
    ]
    for keys, forced_type in group_maps:
        for c in containers:
            for k in keys:
                v = c.get(k)
                if isinstance(v, list):
                    raw_entries.extend([(pm, forced_type) for pm in v])

    # Process and deduplicate
    results = []
    seen = set()  # dedupe by (id, type, masked)
    for pm, forced_type in raw_entries:
        item = _process_pm(pm, forced_type=forced_type)
        if not item:
            continue
        dedupe_key = (item.get("payment_method_id") or "", item.get("method_type") or "", item.get("masked_identifier") or "")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        results.append(item)

    # If still empty, try to infer a single payment method object embedded directly
    if not results:
        # Sometimes a single payment method dict may exist under keys
        single_keys = ("payment_method", "paymentMethod")
        for c in containers:
            for k in single_keys:
                v = c.get(k)
                if isinstance(v, dict):
                    item = _process_pm(v, forced_type=None)
                    if item:
                        dedupe_key = (item.get("payment_method_id") or "", item.get("method_type") or "", item.get("masked_identifier") or "")
                        if dedupe_key not in seen:
                            seen.add(dedupe_key)
                            results.append(item)

    return {"user_id": user_id, "payment_methods": results}
@mcp.tool()
def get_item_details(item_id):
    """
{
  "type": "function",
  "function": {
    "name": "get_item_details",
    "description": "Searches across all product types to locate a specific item by item_id, then returns its parent product_id and product name/type, along with the item's option attributes, current price, availability/stock status, and any compatibility metadata.",
    "parameters": {
      "type": "object",
      "properties": {
        "item_id": {
          "type": "string",
          "description": "The unique item id to look up, such as '1008292230'."
        }
      },
      "required": [
        "item_id"
      ]
    }
  }
}
    """
    item_id = str(item_id).strip()
    if not item_id:
        return {"error": "item_id is required"}

    try:
        product_types = list_all_product_types()
    except Exception as e:
        return {"error": f"Failed to list product types: {e}"}

    if isinstance(product_types, dict) and product_types.get("error"):
        return {"error": f"Failed to list product types: {product_types.get('error')}"}

    # Normalize product types list
    iterables = []
    if isinstance(product_types, list):
        iterables = product_types
    elif isinstance(product_types, dict):
        for key in ["products", "product_types", "items", "data", "results"]:
            if isinstance(product_types.get(key), list):
                iterables = product_types.get(key)
                break
        if not iterables:
            # If the dict itself looks like a single product entry
            if any(k in product_types for k in ["product_id", "productId", "id"]):
                iterables = [product_types]

    if not isinstance(iterables, list) or not iterables:
        # Even if empty, we can't proceed without product ids
        return {"error": "No product types available to search for the given item_id."}

    # Recursive search for an item dict containing the matching item_id

    def _search_item(obj, target):
        if isinstance(obj, dict):
            cand = obj.get("item_id")
            if cand is None and "itemId" in obj:
                cand = obj.get("itemId")
            if cand is not None and str(cand) == str(target):
                return obj
            for v in obj.values():
                found = _search_item(v, target)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for el in obj:
                found = _search_item(el, target)
                if found is not None:
                    return found
        return None

    # Iterate products and search
    for entry in iterables:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("product_id") or entry.get("productId") or entry.get("id")
        if pid is None:
            continue
        pname = entry.get("name") or entry.get("product_name") or entry.get("type") or entry.get("title")
        try:
            details = get_product_details(str(pid))
        except Exception:
            continue
        if isinstance(details, dict) and details.get("error"):
            continue

        found_item = _search_item(details, item_id)
        if not found_item:
            continue

        # Extract product name/type from details if not present
        product_name = pname
        if not product_name and isinstance(details, dict):
            product_name = details.get("name") or details.get("product_name") or details.get("type") or details.get("title")

        # Extract options/attributes
        options = None
        for key in ["options", "attributes", "specs", "option", "variant_options", "variant", "configuration"]:
            if isinstance(found_item.get(key), (dict, list, str, int, float)):
                options = found_item.get(key)
                break
        if options is None:
            common_keys = [
                "size", "color", "colour", "material", "style", "model", "capacity", "length", "width", "height", "dimensions", "switch", "backlight", "connectivity"
            ]
            collected = {}
            for k in common_keys:
                if k in found_item:
                    collected[k] = found_item.get(k)
            if collected:
                options = collected

        # Extract price and currency
        price = None
        currency = None
        val = found_item.get("price")
        if isinstance(val, (int, float, str)):
            price = val
        elif isinstance(val, dict):
            amt = val.get("current") or val.get("amount") or val.get("value") or val.get("price")
            cur = val.get("currency") or val.get("currency_code")
            if amt is not None:
                price = amt
            if cur:
                currency = cur
        if price is None:
            for k in ["current_price", "sale_price", "list_price", "msrp"]:
                if k in found_item:
                    price = found_item.get(k)
                    break
        if price is None and isinstance(found_item.get("pricing"), dict):
            pr = found_item.get("pricing")
            for k in ["current", "price", "sale", "list", "msrp", "amount", "value"]:
                if k in pr:
                    price = pr.get(k)
                    break
            currency = currency or pr.get("currency") or pr.get("currency_code")

        # Extract availability
        availability = None
        if "availability" in found_item:
            availability = found_item.get("availability")
        elif "in_stock" in found_item:
            availability = "in stock" if bool(found_item.get("in_stock")) else "out of stock"
        elif "stock_status" in found_item:
            availability = found_item.get("stock_status")
        elif "available" in found_item:
            availability = "available" if bool(found_item.get("available")) else "unavailable"
        elif "quantity" in found_item:
            q = found_item.get("quantity")
            try:
                qn = int(q)
                availability = "in stock" if qn > 0 else "out of stock"
            except Exception:
                availability = q
        elif "inventory_status" in found_item:
            availability = found_item.get("inventory_status")

        # Extract compatibility metadata
        compatibility = None
        for key in ["compatibility", "compatible_with", "platforms", "ecosystem", "smart_home", "protocols", "supports"]:
            if key in found_item:
                compatibility = found_item.get(key)
                break

        res = {
            "item_id": str(item_id),
            "product_id": str(pid),
            "product_name": product_name,
            "options": options,
            "price": price,
            "currency": currency,
            "availability": availability,
            "compatibility": compatibility
        }
        # Remove None values for conciseness
        cleaned = {k: v for k, v in res.items() if v is not None}
        return cleaned

    return {"error": "Item ID not found"}
if __name__ == "__main__":
    mcp.run(transport='stdio')
