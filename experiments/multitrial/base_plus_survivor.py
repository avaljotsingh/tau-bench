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
@mcp.tool()
def search_items_by_name(query):
    """
{
  "type": "function",
  "function": {
    "name": "search_items_by_name",
    "description": "Searches all product types and their items for names or descriptive text matching the query, and returns matched items with item_id, parent product_id, product name/type, option attributes, current price, and stock availability.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Free-text search, e.g., 'Bright LED Desk Lamp' or 'water bottle size M'."
        }
      },
      "required": [
        "query"
      ]
    }
  }
}
    """
    import re

    q = (query or "").strip()
    if not q:
        return {"error": "Query cannot be empty."}

    # Normalize query and tokens
    q_lower = q.lower()
    tokens = [t for t in re.findall(r"[\w]+", q_lower) if len(t) > 1]

    # Helper to safely stringify fields for matching
    def _flatten_to_text(value):
        try:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, dict):
                parts = []
                for k, v in value.items():
                    parts.append(str(k))
                    parts.append(_flatten_to_text(v))
                return " ".join(parts)
            if isinstance(value, (list, tuple, set)):
                return " ".join(_flatten_to_text(v) for v in value)
            return str(value)
        except Exception:
            return ""

    # Helper to extract items list from a product details payload
    def _extract_items_list(details):
        if not isinstance(details, dict):
            return []
        # direct keys
        for k in ["items", "variants", "skus", "inventory", "variants_list"]:
            v = details.get(k)
            if isinstance(v, list):
                return v
        # nested under 'product'
        prod = details.get("product")
        if isinstance(prod, dict):
            for k in ["items", "variants", "skus", "inventory", "variants_list"]:
                v = prod.get(k)
                if isinstance(v, list):
                    return v
        return []

    # Helper to get product name/type from details or fallback
    def _extract_product_names(details, fallback_name=None):
        pname = None
        ptype = None
        if isinstance(details, dict):
            for k in ["name", "product_name", "title", "display_name"]:
                if details.get(k):
                    pname = details.get(k)
                    break
            for k in ["type", "product_type", "category"]:
                if details.get(k):
                    ptype = details.get(k)
                    break
            # sometimes nested
            if not pname and isinstance(details.get("product"), dict):
                prod = details.get("product")
                for k in ["name", "product_name", "title", "display_name"]:
                    if prod.get(k):
                        pname = prod.get(k)
                        break
                for k in ["type", "product_type", "category"]:
                    if prod.get(k):
                        ptype = prod.get(k)
                        break
        if not pname:
            pname = fallback_name
        return pname, ptype

    # Helper to get item_id across possible keys
    def _extract_item_id(item):
        if not isinstance(item, dict):
            return None
        for k in ["item_id", "id", "sku", "variant_id"]:
            if item.get(k) is not None:
                return str(item.get(k))
        return None

    # Helper to get option attributes structure and text
    def _extract_options(item):
        if not isinstance(item, dict):
            return None, ""
        for k in ["options", "attributes", "option_attributes", "variant_attributes", "specs"]:
            if item.get(k) is not None:
                opts = item.get(k)
                return opts, _flatten_to_text(opts)
        return None, ""

    # Helper to get price in a friendly numeric/string form
    def _extract_price(item):
        if not isinstance(item, dict):
            return None
        if item.get("price") is not None:
            return item.get("price")
        if item.get("current_price") is not None:
            return item.get("current_price")
        if item.get("price_cents") is not None:
            try:
                return round(float(item.get("price_cents")) / 100.0, 2)
            except Exception:
                return item.get("price_cents")
        if item.get("price_minor") is not None:
            try:
                return round(float(item.get("price_minor")) / 100.0, 2)
            except Exception:
                return item.get("price_minor")
        return None

    # Helper to get availability/stock info
    def _extract_availability(item):
        if not isinstance(item, dict):
            return None
        if item.get("availability") is not None:
            return item.get("availability")
        if item.get("in_stock") is not None:
            return "in stock" if bool(item.get("in_stock")) else "out of stock"
        if item.get("stock_status") is not None:
            return item.get("stock_status")
        if item.get("stock") is not None:
            try:
                qty = int(item.get("stock"))
                return f"in stock ({qty})" if qty > 0 else "out of stock"
            except Exception:
                return item.get("stock")
        if item.get("quantity") is not None:
            try:
                qty = int(item.get("quantity"))
                return f"in stock ({qty})" if qty > 0 else "out of stock"
            except Exception:
                return item.get("quantity")
        return None

    # Access catalog: list product types
    try:
        product_types_payload = list_all_product_types()
    except Exception as e:
        return {"error": f"Failed to list product types: {e}"}

    # Normalize product types list
    pt_list = []
    if isinstance(product_types_payload, list):
        pt_list = product_types_payload
    elif isinstance(product_types_payload, dict):
        for key in ["product_types", "products", "items", "data", "results"]:
            if isinstance(product_types_payload.get(key), list):
                pt_list = product_types_payload.get(key)
                break
        if not pt_list and product_types_payload.get("error"):
            return {"error": str(product_types_payload.get("error"))}
    else:
        return {"error": "Unexpected response from list_all_product_types."}

    matches = []

    # Iterate all products and inspect their items
    for pt in pt_list:
        try:
            if not isinstance(pt, dict):
                continue
            pid = pt.get("product_id") or pt.get("id") or pt.get("productId")
            if pid is None:
                continue
            pid = str(pid)
            pt_name = pt.get("name") or pt.get("product_name") or pt.get("type") or pt.get("category")
            # Fetch product details
            try:
                details = get_product_details(product_id=pid)
            except Exception:
                continue
            # Determine product name/type
            product_name, product_type = _extract_product_names(details, fallback_name=pt_name)
            # Gather top-level descriptive text for matching
            top_text_parts = []
            top_text_parts.append(_flatten_to_text(product_name))
            top_text_parts.append(_flatten_to_text(product_type))
            top_text_parts.append(_flatten_to_text(details.get("description") if isinstance(details, dict) else ""))
            if isinstance(details, dict) and isinstance(details.get("product"), dict):
                top_text_parts.append(_flatten_to_text(details.get("product", {}).get("description")))
            top_text = " ".join([t for t in top_text_parts if t]).lower()
            # Iterate items
            items = _extract_items_list(details)
            if not isinstance(items, list):
                continue
            for itm in items:
                if not isinstance(itm, dict):
                    continue
                item_id = _extract_item_id(itm)
                # Item-level text fields
                item_text_parts = []
                for k in ["name", "title", "display_name", "label", "description", "variant_name"]:
                    if itm.get(k):
                        item_text_parts.append(_flatten_to_text(itm.get(k)))
                opts_struct, opts_text = _extract_options(itm)
                if opts_text:
                    item_text_parts.append(opts_text)
                item_text = (top_text + " " + " ".join(item_text_parts)).lower()
                # Matching logic: require all tokens present if tokens exist, else substring match on raw query
                score = 0
                if tokens:
                    token_hits = sum(1 for t in tokens if t in item_text)
                    if token_hits == 0:
                        continue
                    score = token_hits
                else:
                    if q_lower not in item_text:
                        continue
                    score = 1
                # Build result entry
                price = _extract_price(itm)
                availability = _extract_availability(itm)
                entry = {
                    "item_id": item_id,
                    "product_id": pid,
                    "product_name": product_name,
                    "product_type": product_type,
                    "option_attributes": opts_struct,
                    "price": price,
                    "availability": availability,
                }
                # Include minimal additional context if helpful
                # e.g., item name/title if available
                for k in ["name", "title", "display_name", "label", "variant_name"]:
                    if itm.get(k):
                        entry["item_title"] = itm.get(k)
                        break
                entry["_score"] = score
                matches.append(entry)
        except Exception:
            continue

    # Sort matches by score (desc), then by presence of availability (in-stock first if discernible)
    def _availability_rank(av):
        s = str(av).lower()
        if "in stock" in s or s == "true":
            return 0
        if "preorder" in s or "backorder" in s:
            return 1
        if "out of stock" in s or s == "false":
            return 2
        return 1

    matches.sort(key=lambda m: (-int(m.get("_score", 0)), _availability_rank(m.get("availability"))))
    for m in matches:
        if "_score" in m:
            del m["_score"]

    return {"query": q, "items": matches}
@mcp.tool()
def get_order_payment_details(order_id):
    """
{
  "type": "function",
  "function": {
    "name": "get_order_payment_details",
    "description": "Looks up an order and extracts the original payment method used, returning its type (gift card, credit card, PayPal), masked identifier (e.g., last four digits or email), payment_method_id, and a refund timeline hint.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "description": "The order id to look up, such as '#W0000000'. If missing the leading '#', it will be added."
        }
      },
      "required": [
        "order_id"
      ]
    }
  }
}
    """
    import re

    order_id = '' if order_id is None else str(order_id).strip()
    if not order_id:
        return {"error": "order_id is required"}
    if not order_id.startswith('#'):
        order_id = '#' + order_id

    resp = get_order_details(order_id=order_id)
    if not isinstance(resp, dict):
        return {"error": "Unexpected response from get_order_details"}
    if 'error' in resp:
        try:
            return {"error": str(resp['error'])}
        except Exception:
            return {"error": "Failed to fetch order details"}

    order = resp.get('order') or resp.get('data') or resp

    def get_in(d, path):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    # Extract payment_method_id
    candidates_pm_id = [
        ["payment_method_id"], ["paymentMethodId"],
        ["payment", "payment_method_id"], ["payment", "paymentMethodId"],
        ["payment_method", "id"], ["paymentDetails", "id"],
        ["payment", "id"], ["payment_info", "id"], ["paymentInfo", "id"]
    ]
    pm_id = None
    for p in candidates_pm_id:
        val = get_in(order, p)
        if isinstance(val, str) and val.strip():
            pm_id = val.strip()
            break

    if pm_id is None:
        charges = order.get('charges') if isinstance(order, dict) else None
        if isinstance(charges, list) and charges:
            for ch in charges:
                if isinstance(ch, dict):
                    val = ch.get('payment_method_id') or ch.get('paymentMethodId') or ch.get('payment_method') or ch.get('id')
                    if isinstance(val, str) and val.strip():
                        pm_id = val.strip()
                        break

    if pm_id is None:
        return {"error": "Payment method information not found for this order."}

    # Extract payment method type
    candidates_type = [
        ["payment_method_type"], ["paymentType"],
        ["payment", "payment_method_type"], ["payment", "type"],
        ["payment_method", "type"], ["paymentDetails", "type"],
        ["payment_info", "type"], ["paymentInfo", "type"]
    ]
    p_type = None
    for p in candidates_type:
        val = get_in(order, p)
        if isinstance(val, str) and val.strip():
            p_type = val.strip()
            break

    if not p_type:
        lid = pm_id.lower()
        if lid.startswith('gift_card') or ('gift' in lid and 'card' in lid):
            p_type = 'gift card'
        elif lid.startswith('paypal') or 'paypal' in lid:
            p_type = 'PayPal'
        elif lid.startswith('credit_card') or 'card' in lid:
            p_type = 'credit card'

    # Extract masked identifier
    masked = None
    candidates_mask = [
        ["masked_identifier"], ["masked"], ["mask"],
        ["payment", "masked_identifier"], ["payment", "masked"],
        ["payment_method", "masked_identifier"], ["payment_method", "masked"],
        ["paymentDetails", "masked_identifier"], ["paymentDetails", "masked"],
        ["payment_info", "masked_identifier"], ["payment_info", "masked"],
        ["paymentInfo", "masked_identifier"], ["paymentInfo", "masked"],
        ["payment", "display"], ["paymentDetails", "display"], ["payment_method", "display"]
    ]
    for p in candidates_mask:
        val = get_in(order, p)
        if isinstance(val, str) and val.strip():
            masked = val.strip()
            break

    # Try last4-based masking
    if not masked:
        last4 = None
        last4_keys = [["last4"], ["last_four"], ["ending_in"], ["suffix"], ["payment", "last4"], ["payment_method", "last4"], ["paymentDetails", "last4"], ["payment_info", "last4"], ["paymentInfo", "last4"]]
        for p in last4_keys:
            val = get_in(order, p)
            if isinstance(val, (str, int)):
                s = str(val).strip()
                if s.isdigit():
                    last4 = s[-4:]
                    break
        if last4:
            masked = '\u2022\u2022\u2022\u2022 ' + last4

    # Try email
    if not masked:
        email_keys = [["email"], ["paypal_email"], ["payment", "email"], ["paymentDetails", "email"], ["payment_method", "email"], ["payment_info", "email"], ["paymentInfo", "email"]]
        for p in email_keys:
            val = get_in(order, p)
            if isinstance(val, str) and '@' in val:
                masked = val.strip()
                break

    # Derive from pm_id if still missing
    if not masked:
        if p_type and p_type.lower() == 'credit card':
            m = re.search(r"(\d{4})\b", pm_id)
            if m:
                masked = '\u2022\u2022\u2022\u2022 ' + m.group(1)
        elif p_type and p_type.lower() == 'paypal':
            m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", pm_id)
            if m:
                masked = m.group(1)
        elif p_type and p_type.lower() == 'gift card':
            m = re.search(r"(\d{4})\b", pm_id)
            if m:
                masked = '\u2022\u2022\u2022\u2022 ' + m.group(1)

    # Refund timeline hint
    timeline = 'immediate' if (p_type and p_type.lower() == 'gift card') else '5-7 business days'

    return {
        "order_id": order_id,
        "payment_method_id": pm_id,
        "type": p_type or 'unknown',
        "masked_identifier": masked or None,
        "refund_timeline": timeline
    }
@mcp.tool()
def list_order_items(order_id):
    """
{
  "type": "function",
  "function": {
    "name": "list_order_items",
    "description": "Fetches an order by order_id (auto-prefixing '#' if missing) and compiles a detailed list of its items with item_id, parent product_id, product name/type, option attributes, quantity, and current price by enriching each item via get_item_details.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "description": "The order id, such as '#W0000000'. If missing the leading '#', it will be added automatically."
        }
      },
      "required": [
        "order_id"
      ]
    }
  }
}
    """
    order_id = (order_id or "").strip()
    if not order_id:
        return {"error": "order_id is required"}
    if not order_id.startswith("#"):
        order_id = "#" + order_id
    try:
        order = get_order_details(order_id=order_id)
    except Exception as e:
        return {"error": f"failed to fetch order details: {e}"}
    if not isinstance(order, dict):
        return {"error": "unexpected response format from get_order_details"}
    if "error" in order:
        return order

    def _find_items_in_structure(obj):
        # Recursively search for a list that looks like order items
        if isinstance(obj, dict):
            # Prefer common keys first
            for key in ["items", "order_items", "line_items", "products"]:
                v = obj.get(key)
                if isinstance(v, list) and len(v) > 0:
                    return v
            # Fallback: scan any list value
            for k, v in obj.items():
                if isinstance(v, list) and len(v) > 0:
                    return v
            # Recurse into dict values
            for v in obj.values():
                found = _find_items_in_structure(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for x in obj:
                found = _find_items_in_structure(x)
                if found is not None:
                    return found
        return None

    def _extract_item_id(entry):
        if isinstance(entry, str):
            return entry.strip()
        if not isinstance(entry, dict):
            return None
        # Common keys for item id
        for k in ("item_id", "itemId", "id", "sku", "variant_id", "variantId"):
            if k in entry:
                val = entry.get(k)
                if isinstance(val, (str, int, float)):
                    return str(val)
        # Sometimes nested under 'item' or 'variant'
        for parent in ("item", "variant"):
            if parent in entry and isinstance(entry[parent], dict):
                for k in ("item_id", "itemId", "id", "sku", "variant_id", "variantId"):
                    if k in entry[parent]:
                        val = entry[parent].get(k)
                        if isinstance(val, (str, int, float)):
                            return str(val)
        return None

    def _extract_quantity(entry):
        if isinstance(entry, dict):
            for k in ("quantity", "qty", "count", "units", "amount"):
                if k in entry:
                    q = entry.get(k)
                    try:
                        return int(q)
                    except Exception:
                        try:
                            return int(float(str(q)))
                        except Exception:
                            return 1
        return 1

    items_list = _find_items_in_structure(order)
    if not isinstance(items_list, list):
        # Could not locate items list; return gracefully
        return {"order_id": order_id, "items": [], "warning": "No items found in order details"}

    # Aggregate quantities per item_id
    item_quantities = {}
    for ent in items_list:
        iid = _extract_item_id(ent)
        if not iid:
            # skip entries without clear item_id
            continue
        q = _extract_quantity(ent)
        item_quantities[iid] = item_quantities.get(iid, 0) + (q if isinstance(q, int) else 1)

    results = []
    for iid, qty in item_quantities.items():
        try:
            det = get_item_details(item_id=iid)
        except Exception as e:
            results.append({
                "item_id": iid,
                "quantity": qty,
                "error": f"failed to fetch item details: {e}"
            })
            continue
        if not isinstance(det, dict):
            results.append({"item_id": iid, "quantity": qty, "error": "unexpected item details format"})
            continue
        if "error" in det:
            results.append({"item_id": iid, "quantity": qty, "error": det.get("error")})
            continue
        # Extract fields with fallbacks
        product_id = det.get("product_id") or det.get("productId") or det.get("parent_product_id") or det.get("parentProductId")
        product_name = det.get("product_name") or det.get("name") or det.get("product_title") or det.get("title")
        product_type = det.get("product_type") or det.get("type")
        options = det.get("option_attributes") or det.get("options") or det.get("attributes") or det.get("specs")
        # Current price
        current_price = None
        if isinstance(det.get("current_price"), (int, float, str)):
            current_price = det.get("current_price")
        elif isinstance(det.get("price"), (int, float, str)):
            current_price = det.get("price")
        elif isinstance(det.get("unit_price"), (int, float, str)):
            current_price = det.get("unit_price")
        else:
            pricing = det.get("pricing") if isinstance(det.get("pricing"), dict) else None
            if pricing:
                for k in ("current_price", "price", "unit_price"):
                    if k in pricing and isinstance(pricing.get(k), (int, float, str)):
                        current_price = pricing.get(k)
                        break

        results.append({
            "item_id": iid,
            "product_id": product_id,
            "product_name": product_name,
            "product_type": product_type,
            "options": options,
            "quantity": qty,
            "current_price": current_price
        })

    return {"order_id": order_id, "items": results}
@mcp.tool()
def list_product_variants(product_id):
    """
{
  "type": "function",
  "function": {
    "name": "list_product_variants",
    "description": "Lists all variant items for a given product_id, returning each variant's item_id, option attributes (e.g., size, color), current price, and stock availability. Useful for locating specific variants, checking availability, identifying the cheapest option, and collecting new_item_ids for modify/exchange actions.",
    "parameters": {
      "type": "object",
      "properties": {
        "product_id": {
          "type": "string",
          "description": "The product id whose variants to list, e.g., '6086499569'."
        }
      },
      "required": [
        "product_id"
      ]
    }
  }
}
    """
    import math
    import json

    # Validate input
    if product_id is None or not isinstance(product_id, str) or not product_id.strip():
        return {"error": "product_id must be a non-empty string"}

    # Helper parsers
    def _to_float_price(obj):
        # Try common fields for price
        for key in ("price", "current_price", "sale_price", "amount", "unit_price"):
            if isinstance(obj, dict) and key in obj and obj[key] is not None:
                try:
                    return float(obj[key])
                except Exception:
                    pass
        # Cents-based fields
        for key in ("price_cents", "amount_cents"):
            if isinstance(obj, dict) and key in obj and obj[key] is not None:
                try:
                    return float(obj[key]) / 100.0
                except Exception:
                    pass
        return None

    def _extract_options(obj):
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("options"), dict):
            return obj.get("options")
        if isinstance(obj.get("attributes"), dict):
            return obj.get("attributes")
        if isinstance(obj.get("option_attributes"), dict):
            return obj.get("option_attributes")
        # Heuristic: collect common option keys if present
        option_keys = [
            "size", "color", "colour", "width", "length", "style", "material", "capacity",
            "pattern", "fit", "waist", "inseam", "variant", "finish"
        ]
        found = {k: obj[k] for k in option_keys if k in obj}
        return found if found else None


    def _extract_availability(obj):
        # Returns (in_stock_bool_or_None, stock_int_or_None)
        if not isinstance(obj, dict):
            return (None, None)
        # stock count
        stock = None
        for k in ("stock", "inventory", "quantity_available", "qty", "available_quantity"):
            if k in obj and obj[k] is not None:
                try:
                    stock = int(obj[k])
                except Exception:
                    # if boolean sneaks in, ignore
                    pass
                break
        # booleans
        in_stock = None
        for k in ("in_stock", "available", "is_available", "is_in_stock"):
            if k in obj and isinstance(obj[k], bool):
                in_stock = obj[k]
                break
        # availability string
        if in_stock is None and isinstance(obj.get("availability"), str):
            s = obj.get("availability").strip().lower()
            if any(tag in s for tag in ["in stock", "available", "instock"]):
                in_stock = True
            elif any(tag in s for tag in ["out of stock", "unavailable", "sold out"]):
                in_stock = False
        if in_stock is None and isinstance(stock, int):
            in_stock = stock > 0
        return (in_stock, stock)


    def _extract_item_id(obj):
        if isinstance(obj, str):
            return obj
        if not isinstance(obj, dict):
            return None
        for k in ("item_id", "itemId", "id", "sku_id", "sku", "variant_id"):
            if k in obj and obj[k] is not None:
                try:
                    return str(obj[k])
                except Exception:
                    return None
        return None


    def _enrich_from_get_item_details(item_id):
        try:
            d = get_item_details(item_id=item_id)
        except Exception as e:
            return {}
        # Parse JSON string responses
        if isinstance(d, str):
            try:
                d = json.loads(d)
            except Exception:
                return {}
        if not isinstance(d, dict):
            return {}
        # Some APIs may nest details under a key; flatten best-effort
        candidate = d
        # If there's a single nested dict containing options/price fields, use it
        for key in ("item", "data", "details", "result"):
            if isinstance(d.get(key), dict):
                candidate = d.get(key)
                break
        price = _to_float_price(candidate)
        opts = _extract_options(candidate)
        instock, stock = _extract_availability(candidate)
        return {
            "price": price,
            "options": opts,
            "in_stock": instock,
            "stock": stock,
            "product_id": candidate.get("product_id") or d.get("product_id"),
            "product_name": candidate.get("product_name") or candidate.get("product") or d.get("product_name") or d.get("product")
        }


    # Fetch product details
    try:
        p = get_product_details(product_id=product_id)
    except Exception as e:
        return {"error": f"get_product_details failed: {e}"}

    # Parse JSON string response if needed
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:
            return {"error": "Unexpected response format from get_product_details"}

    if not isinstance(p, dict):
        return {"error": "Unexpected response format from get_product_details"}
    if "error" in p:
        return {"error": p.get("error")}

    product_name = p.get("name") or p.get("product_name") or p.get("title")

    # Collect potential variant containers
    candidates = []
    for key in ("items", "variants", "skus", "inventory", "inventory_items", "children"):
        v = p.get(key)
        if isinstance(v, list):
            candidates.extend(v)
        elif isinstance(v, dict):
            # If dict of item_id -> details
            candidates.extend(list(v.values()))

    # If still empty, sometimes product details contain top-level 'item_ids'
    if not candidates and isinstance(p.get("item_ids"), list):
        candidates.extend(p.get("item_ids"))

    if not candidates:
        return {
            "product_id": product_id,
            "product_name": product_name,
            "variants": []
        }

    variants = []
    seen = set()
    for cand in candidates:
        iid = _extract_item_id(cand)
        if not iid or iid in seen:
            continue
        seen.add(iid)
        # Try to extract from candidate directly first
        price = _to_float_price(cand) if isinstance(cand, dict) else None
        options = _extract_options(cand) if isinstance(cand, dict) else None
        in_stock, stock = _extract_availability(cand) if isinstance(cand, dict) else (None, None)

        # Enrich missing fields via get_item_details
        if price is None or options is None or in_stock is None:
            enriched = _enrich_from_get_item_details(iid)
            if price is None:
                price = enriched.get("price")
            if options is None:
                options = enriched.get("options")
            if in_stock is None:
                in_stock = enriched.get("in_stock")
            if stock is None:
                stock = enriched.get("stock")
            if product_name is None:
                product_name = enriched.get("product_name") or product_name

        # Ensure JSON-serializable primitives
        try:
            price_val = float(price) if price is not None and not (isinstance(price, float) and (math.isnan(price) or math.isinf(price))) else price
        except Exception:
            price_val = None

        variant_entry = {
            "item_id": str(iid),
            "options": options if isinstance(options, dict) else ({} if options is None else {"value": options}),
            "price": price_val,
            "in_stock": bool(in_stock) if isinstance(in_stock, bool) else (None if in_stock is None else bool(in_stock)),
            "stock": stock if (isinstance(stock, int) or stock is None) else None
        }
        variants.append(variant_entry)

    # Sort by price (ascending) when available, otherwise by item_id
    variants.sort(key=lambda x: (float('inf') if x.get('price') is None else x.get('price'), x.get('item_id')))

    return {
        "product_id": product_id,
        "product_name": product_name,
        "variants": variants
    }

@mcp.tool()
def find_order_by_item_with_tracking(user_id, item_query):
    """
{
  "type": "function",
  "function": {
    "name": "find_order_by_item_with_tracking",
    "description": "Finds a user\u2019s order that contains an item matching a free-text query (e.g., 'tablet'), robustly handling flaky get_order_details by retrying with backoff and falling back to list_order_items for matching, then re-attempting details to extract tracking. Returns order_id, status, matched item_ids, and tracking_ids.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string",
          "description": "The user id to search orders for, e.g., 'olivia_lopez_3865'."
        },
        "item_query": {
          "type": "string",
          "description": "Minimal hint to identify the item, e.g., 'tablet' or 'laptop 13-inch'."
        }
      },
      "required": [
        "user_id",
        "item_query"
      ]
    }
  }
}
    """
    import json, time

    # Validate inputs
    if user_id is None or str(user_id).strip() == "":
        return {"error": "user_id is required"}
    if item_query is None or str(item_query).strip() == "":
        return {"error": "item_query is required"}

    item_query_lc = str(item_query).strip().lower()

    # Helpers

    def _ensure_hash(oid):
        if not isinstance(oid, str):
            oid = str(oid)
        oid = oid.strip()
        if not oid:
            return oid
        return oid if oid.startswith('#') else '#' + oid


    def _parse_tool_result(res):
        # Tools may return dict or JSON string or an error string
        if isinstance(res, dict):
            return res, None
        if isinstance(res, str):
            s = res.strip()
            # detect explicit error format
            if s.lower().startswith("error:"):
                return None, s
            # try to parse JSON
            try:
                return json.loads(s), None
            except Exception as e:
                return None, f"Unparseable tool output: {str(e)}"
        # Unexpected type
        try:
            return json.loads(str(res)), None
        except Exception as e:
            return None, f"Unexpected tool output type: {type(res).__name__}"


    def _call_get_order_details_with_retries(order_id, attempts=4):
        order_id = _ensure_hash(order_id)
        delay = 0.05
        last_err = None
        for i in range(max(1, int(attempts))):
            try:
                r = get_order_details(order_id=order_id)
            except Exception as e:
                last_err = f"Exception calling get_order_details: {str(e)}"
                time.sleep(delay)
                delay = min(delay * 2, 0.8)
                continue
            data, err = _parse_tool_result(r)
            if err is None and isinstance(data, dict) and data.get("order_id"):
                return data, None
            last_err = err if err else "get_order_details returned invalid data"
            time.sleep(delay)
            delay = min(delay * 2, 0.8)
        return None, last_err or "get_order_details failed after retries"


    def _call_list_order_items(order_id):
        order_id = _ensure_hash(order_id)
        try:
            r = list_order_items(order_id=order_id)
        except Exception as e:
            return None, f"Exception calling list_order_items: {str(e)}"
        data, err = _parse_tool_result(r)
        if err:
            return None, err
        return data, None


    def _text_candidates_from_item(it):
        texts = []
        # Common fields
        for k in ("name", "product_name", "product_type", "title", "category", "type"):
            v = it.get(k)
            if isinstance(v, str):
                texts.append(v)
        # Options/attributes
        opts = it.get("options")
        if isinstance(opts, dict):
            for v in opts.values():
                try:
                    texts.append(str(v))
                except Exception:
                    pass
        return [t for t in texts if isinstance(t, str)]


    def _match_items(items, query_lc):
        matched_ids = []
        if not isinstance(items, list):
            return matched_ids
        for it in items:
            if not isinstance(it, dict):
                continue
            texts = _text_candidates_from_item(it)
            # If no common fields found, also try the whole item as string (last resort)
            blob = " ".join(texts).lower()
            if query_lc in blob:
                iid = it.get("item_id")
                # In get_order_details, item_id should exist; if missing, skip
                if iid is None:
                    # try alternative keys
                    iid = it.get("id")
                if iid is not None:
                    matched_ids.append(str(iid))
        return matched_ids


    def _extract_tracking_for_items(details, matched_item_ids):
        # Extract tracking_ids for any fulfillment touching the matched item_ids; if none, aggregate all
        trks = []
        if not isinstance(details, dict):
            return trks
        fulf = details.get("fulfillments")
        if isinstance(fulf, list):
            any_linked = False
            for f in fulf:
                if not isinstance(f, dict):
                    continue
                f_item_ids = f.get("item_ids")
                f_trk = f.get("tracking_id")
                # Normalize tracking to list
                f_trk_list = []
                if isinstance(f_trk, list):
                    f_trk_list = [str(x) for x in f_trk]
                elif isinstance(f_trk, (str, int)):
                    f_trk_list = [str(f_trk)]
                # Check link
                linked = False
                if isinstance(f_item_ids, list) and matched_item_ids:
                    for iid in f_item_ids:
                        if str(iid) in set(matched_item_ids):
                            linked = True
                            break
                if linked:
                    any_linked = True
                    for t in f_trk_list:
                        if t not in trks:
                            trks.append(t)
            # If no linked fulfillment found but fulfillments exist, include all tracking ids
            if not trks and isinstance(fulf, list):
                for f in fulf:
                    if not isinstance(f, dict):
                        continue
                    f_trk = f.get("tracking_id")
                    if isinstance(f_trk, list):
                        for t in f_trk:
                            t = str(t)
                            if t not in trks:
                                trks.append(t)
                    elif isinstance(f_trk, (str, int)):
                        t = str(f_trk)
                        if t not in trks:
                            trks.append(t)
        return trks

    # 1) Fetch user's orders
    try:
        ud_res = get_user_details(user_id=user_id)
    except Exception as e:
        return {"error": f"Failed to get user details: {str(e)}"}
    ud, ud_err = _parse_tool_result(ud_res)
    if ud_err:
        return {"error": f"Failed to get user details: {ud_err}"}
    orders = ud.get("orders") if isinstance(ud, dict) else None
    if not isinstance(orders, list) or not orders:
        return {"error": "No orders found for user or invalid orders list"}

    # Iterate through orders to find a match
    for raw_oid in orders:
        oid = _ensure_hash(raw_oid)
        # 2) Try get_order_details with retries
        details, derr = _call_get_order_details_with_retries(oid, attempts=4)
        if details:
            items = details.get("items")
            matched = _match_items(items, item_query_lc)
            if matched:
                status = details.get("status")
                tracking_ids = _extract_tracking_for_items(details, matched)
                return {
                    "order_id": oid,
                    "status": status,
                    "matched_item_ids": matched,
                    "tracking_ids": tracking_ids
                }
        # 3) If details failed or no match, fall back to list_order_items to check for match
        loi, loi_err = _call_list_order_items(oid)
        if loi:
            loi_items = loi.get("items") if isinstance(loi, dict) else None
            matched_from_list = _match_items(loi_items, item_query_lc)
            if matched_from_list:
                # 4) Re-attempt get_order_details to extract fulfillments/tracking
                details2, derr2 = _call_get_order_details_with_retries(oid, attempts=4)
                if details2:
                    status = details2.get("status")
                    tracking_ids = _extract_tracking_for_items(details2, matched_from_list)
                    return {
                        "order_id": oid,
                        "status": status,
                        "matched_item_ids": matched_from_list,
                        "tracking_ids": tracking_ids
                    }
                else:
                    # Could not retrieve tracking even after reattempts; return partial with empty tracking
                    return {
                        "order_id": oid,
                        "status": None,
                        "matched_item_ids": matched_from_list,
                        "tracking_ids": []
                    }
    # If not found in any order
    return {"error": "No order found containing an item matching the query"}

if __name__ == "__main__":
    mcp.run(transport='stdio')
