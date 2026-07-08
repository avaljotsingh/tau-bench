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
def authenticate_user(email, first_name, last_name, zip, provided_user_id):
    """
{
  "type": "function",
  "function": {
    "name": "authenticate_user",
    "description": "Authenticates a user by locating their user_id, preferring email lookup and falling back to first_name+last_name+zip. If provided_user_id is given, checks and reports whether it matches the located account.",
    "parameters": {
      "type": "object",
      "properties": {
        "email": {
          "type": "string",
          "description": "User's email address to look up, e.g., 'user@example.com'."
        },
        "first_name": {
          "type": "string",
          "description": "User's first name to use with last name and zip if email lookup is unavailable."
        },
        "last_name": {
          "type": "string",
          "description": "User's last name to use with first name and zip if email lookup is unavailable."
        },
        "zip": {
          "type": "string",
          "description": "User's zip code to use with first and last name if email lookup is unavailable."
        },
        "provided_user_id": {
          "type": "string",
          "description": "A user_id supplied by the user; will be checked for consistency with the located account."
        }
      },
      "required": []
    }
  }
}
    """
    email = (email or '').strip()
    first_name = (first_name or '').strip()
    last_name = (last_name or '').strip()
    zip = (zip or '').strip()
    provided_user_id = (provided_user_id or '').strip()

    # Validate input: need email or full name+zip
    if not email and not (first_name and last_name and zip):
        return {
            'error': 'Please provide either an email, or first name + last name + zip to locate your account.'
        }

    # Helper to robustly extract user_id from tool responses
    def _extract_user_id(resp):
        # Returns (user_id, error_message)
        if isinstance(resp, dict):
            if 'error' in resp and resp.get('error'):
                return None, str(resp.get('error'))
            # Common key patterns
            for key in ['user_id', 'id', 'uid', 'userId']:
                v = resp.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip(), None
            # Nested under 'data'
            data = resp.get('data') if isinstance(resp.get('data'), dict) else None
            if data:
                for key in ['user_id', 'id', 'uid', 'userId']:
                    v = data.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip(), None
            # If dict has exactly one string value, consider it
            try:
                values = [v for v in resp.values() if isinstance(v, str) and v.strip()]
                if len(values) == 1:
                    return values[0].strip(), None
            except Exception:
                pass
            return None, 'Unexpected response format when extracting user id.'
        elif isinstance(resp, str):
            s = resp.strip()
            return (s, None) if s else (None, 'Empty response string from lookup.')
        else:
            return None, 'Unexpected response type from lookup.'

    user_id = None
    found_via = None
    attempts = []
    last_error = None

    # Try email first if provided
    if email:
        try:
            resp = find_user_id_by_email(email=email)
        except Exception as e:
            resp = {'error': f'Lookup by email failed: {e}'}
        uid, err = _extract_user_id(resp)
        attempts.append({'method': 'email', 'success': bool(uid), 'error': err})
        if uid:
            user_id = uid
            found_via = 'email'

    # Fallback to name+zip if needed and available
    if not user_id and first_name and last_name and zip:
        try:
            resp = find_user_id_by_name_zip(first_name=first_name, last_name=last_name, zip=zip)
        except Exception as e:
            resp = {'error': f'Lookup by name+zip failed: {e}'}
        uid, err = _extract_user_id(resp)
        attempts.append({'method': 'name_zip', 'success': bool(uid), 'error': err})
        if uid:
            user_id = uid
            found_via = 'name_zip'
        else:
            last_error = err

    if not user_id:
        # Compose a concise error message
        if attempts:
            # Prefer the most recent error with details
            detailed_errors = [a['error'] for a in attempts if a.get('error')]
            msg = detailed_errors[-1] if detailed_errors else 'Unable to locate user with provided information.'
        else:
            msg = 'Unable to locate user with provided information.'
        return {
            'error': msg,
            'attempts': attempts
        }

    # Check provided_user_id consistency
    match = None
    if provided_user_id:
        match = (provided_user_id == user_id)

    result = {
        'authenticated': True,
        'user_id': user_id,
        'found_via': found_via,
        'provided_user_id': provided_user_id or None,
        'provided_user_id_matches': match
    }

    # Optionally include which inputs were used
    if found_via == 'email':
        result['email_used'] = email
    elif found_via == 'name_zip':
        result['name_zip_used'] = {
            'first_name': first_name,
            'last_name': last_name,
            'zip': zip
        }

    return result


@mcp.tool()
def resolve_user_id(email, first_name, last_name, zip):
    """
{
  "type": "function",
  "function": {
    "name": "resolve_user_id",
    "description": "Resolves and returns a user's ID by first attempting email lookup, and if not found or no email provided, falls back to first_name + last_name + zip lookup. Returns {'user_id': ...} on success or {'error': ...} on failure.",
    "parameters": {
      "type": "object",
      "properties": {
        "email": {
          "type": "string",
          "description": "User email (preferred identifier), e.g., 'user@example.com'. Optional."
        },
        "first_name": {
          "type": "string",
          "description": "User first name, e.g., 'John'. Optional, used with last_name and zip if email not provided or not found."
        },
        "last_name": {
          "type": "string",
          "description": "User last name, e.g., 'Doe'. Optional, used with first_name and zip if email not provided or not found."
        },
        "zip": {
          "type": "string",
          "description": "User zip code, e.g., '12345'. Optional, used with first_name and last_name if email not provided or not found."
        }
      },
      "required": []
    }
  }
}
    """
    import json

    # Support calls where the caller bundles all args into the 'email' field as a JSON object string
    try:
        if isinstance(email, str):
            s_email = email.strip()
            if s_email.startswith('{') and s_email.endswith('}'):
                try:
                    payload = json.loads(s_email)
                    if isinstance(payload, dict):
                        email = payload.get('email', email)
                        # Only override if present in payload
                        if 'first_name' in payload:
                            first_name = payload.get('first_name')
                        if 'last_name' in payload:
                            last_name = payload.get('last_name')
                        if 'zip' in payload:
                            zip = payload.get('zip')
                except Exception:
                    pass
    except Exception:
        pass

    def _clean(s):
        if s is None:
            return ''
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return ''
        s2 = s.strip()
        if s2.lower() in ('', 'null', 'none', 'nil', 'n/a'):
            return ''
        return s2

    def _extract_user_id(resp):
        keys = ['user_id', 'id', 'userId', 'uid']
        def _search(val):
            if isinstance(val, dict):
                # If explicit error key, treat as not found
                if 'error' in val and isinstance(val['error'], str) and val['error'].strip():
                    return None
                for k in keys:
                    if k in val and isinstance(val[k], str) and val[k].strip():
                        return val[k].strip()
                for v in val.values():
                    found = _search(v)
                    if found:
                        return found
            elif isinstance(val, list):
                for v in val:
                    found = _search(v)
                    if found:
                        return found
            elif isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                # Heuristics: do not treat obvious error messages as user ids
                low = s.lower()
                if low.startswith('error') or 'not found' in low or 'no user' in low or 'unable to' in low:
                    return None
                # Also handle JSON-encoded responses
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        return _search(json.loads(s))
                    except Exception:
                        return None
                return s
            return None
        return _search(resp)

    email_c = _clean(email)
    first_c = _clean(first_name)
    last_c = _clean(last_name)
    zip_c = _clean(zip)

    # Try email first if provided
    if email_c:
        try:
            resp = find_user_id_by_email(email=email_c)
            uid = _extract_user_id(resp)
            if uid:
                return {"user_id": uid}
        except Exception:
            # proceed to fallback
            pass

    # Fallback to name + zip if possible
    if first_c and last_c and zip_c:
        try:
            resp2 = find_user_id_by_name_zip(first_name=first_c, last_name=last_c, zip=zip_c)
            uid2 = _extract_user_id(resp2)
            if uid2:
                return {"user_id": uid2}
            else:
                return {"error": "User not found with provided identifiers."}
        except Exception as e:
            return {"error": f"Lookup by name and zip failed: {str(e)}"}

    # If we reach here, either email was not provided/found and we lack sufficient fallback info
    if email_c:
        return {"error": "User not found via email, and insufficient name+zip provided for fallback. Please provide first_name, last_name, and zip."}
    else:
        return {"error": "Please provide either an email, or first_name + last_name + zip to locate the user."}


@mcp.tool()
def list_user_orders(user_id, status, since, until, limit, sort):
    """
{
  "type": "function",
  "function": {
    "name": "list_user_orders",
    "description": "Fetches a user's orders and compiles a concise, filterable, and sortable list. For each order, returns order_id (with leading '#'), placed_at (EST, 24h), status, total, item_count, and a compact items summary (product name/type plus key options). Supports filtering by status and date range (since/until), sort by newest/oldest, and optional limit.",
    "parameters": {
      "type": "object",
      "properties": {
        "user_id": {
          "type": "string",
          "description": "The user id to look up, such as 'sara_doe_496'."
        },
        "status": {
          "type": "string",
          "description": "Optional; one of 'pending', 'processed', 'delivered', 'cancelled', or 'any'. Defaults to 'any'."
        },
        "since": {
          "type": "string",
          "description": "Optional; lower bound timestamp in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (EST). Inclusive."
        },
        "until": {
          "type": "string",
          "description": "Optional; upper bound timestamp in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (EST). Inclusive."
        },
        "limit": {
          "type": "string",
          "description": "Optional; maximum number of orders to return, integer as string."
        },
        "sort": {
          "type": "string",
          "description": "Optional; 'newest' (default) or 'oldest'."
        }
      },
      "required": [
        "user_id"
      ]
    }
  }
}
    """
    import json
    from datetime import datetime

    # Compatibility shim: allow a single dict payload to be passed via user_id
    # and unpack optional fields if others are empty/missing
    try:
        _payload_is_dict = isinstance(user_id, dict)
    except Exception:
        _payload_is_dict = False
    if _payload_is_dict and not (status or since or until or limit or sort):
        _p = user_id
        user_id = _p.get("user_id", "")
        status = _p.get("status", "")
        since = _p.get("since", "")
        until = _p.get("until", "")
        # Ensure limit remains a string if present
        _lim = _p.get("limit", "")
        if _lim is None:
            limit = ""
        else:
            limit = str(_lim)
        sort = _p.get("sort", "")

    # Ensure optional params exist
    try:
        status
    except NameError:
        status = ""
    try:
        since
    except NameError:
        since = ""
    try:
        until
    except NameError:
        until = ""
    try:
        limit
    except NameError:
        limit = ""
    try:
        sort
    except NameError:
        sort = ""

    # Validate user_id
    if not isinstance(user_id, str) or not user_id.strip():
        return {"error": "user_id is required and must be a non-empty string."}
    user_id = user_id.strip()

    # Normalize and validate status
    valid_statuses = {"pending", "processed", "delivered", "cancelled", "any", ""}
    status_norm = status.strip().lower() if isinstance(status, str) else ""
    if not status_norm:
        status_norm = "any"
    if status_norm not in valid_statuses:
        return {"error": "Invalid status; must be one of: pending, processed, delivered, cancelled, any."}

    # Normalize and validate sort
    sort_norm = sort.strip().lower() if isinstance(sort, str) else "newest"
    if not sort_norm:
        sort_norm = "newest"
    if sort_norm not in ("newest", "oldest"):
        return {"error": "Invalid sort; must be 'newest' or 'oldest'."}

    # Parse limit
    lim = None
    if isinstance(limit, str) and limit.strip():
        try:
            lim = int(limit.strip())
            if lim < 0:
                return {"error": "limit must be a non-negative integer."}
        except Exception:
            return {"error": "limit must be an integer."}

    # Date parsing helper
    def _parse_dt(s):
        if not s:
            return None
        s = str(s).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                pass
        return None

    since_dt = None
    if isinstance(since, str) and since.strip():
        since_dt = _parse_dt(since)
        if since_dt is None:
            return {"error": "Invalid 'since' datetime; use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (EST)."}

    until_dt = None
    if isinstance(until, str) and until.strip():
        until_dt = _parse_dt(until)
        if until_dt is None:
            return {"error": "Invalid 'until' datetime; use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' (EST)."}

    # Fetch user details
    try:
        ud = get_user_details(user_id=user_id)
    except Exception as e:
        return {"error": f"get_user_details failed: {str(e)}"}

    # Parse JSON string responses from tools
    if isinstance(ud, str):
        try:
            ud = json.loads(ud)
        except Exception:
            # If not JSON, leave as-is
            pass

    if isinstance(ud, dict) and ud.get("error"):
        return {"error": ud.get("error")}

    # Locate orders list from various possible keys
    orders_raw = None
    if isinstance(ud, dict):
        for k in ["orders", "order_history", "recent_orders", "order_ids", "orderIds", "orders_list"]:
            if k in ud:
                orders_raw = ud.get(k)
                break
        if orders_raw is None and isinstance(ud.get("user"), dict):
            u = ud.get("user")
            for k in ["orders", "order_history", "recent_orders", "order_ids", "orderIds", "orders_list"]:
                if k in u:
                    orders_raw = u.get(k)
                    break

    if not orders_raw:
        return {
            "user_id": user_id,
            "filters": {"status": status_norm, "since": since, "until": until, "limit": lim, "sort": sort_norm},
            "count": 0,
            "orders": []
        }

    # Ensure iterable
    if isinstance(orders_raw, dict):
        orders_iterable = list(orders_raw.values())
    else:
        orders_iterable = list(orders_raw) if isinstance(orders_raw, list) else []

    results = []

    for entry in orders_iterable:
        # Extract order_id
        oid_raw = None
        entry_meta = {}
        if isinstance(entry, str):
            oid_raw = entry
        elif isinstance(entry, dict):
            entry_meta = entry
            oid_raw = entry.get("order_id") or entry.get("id") or entry.get("orderId")
        else:
            continue

        if not oid_raw:
            continue
        oid = str(oid_raw)
        if not oid.startswith("#"):
            oid = f"#{oid}"

        # Fetch order details
        try:
            od = get_order_details(order_id=oid)
        except Exception as e:
            od = {"error": str(e)}

        if isinstance(od, str):
            try:
                od = json.loads(od)
            except Exception:
                od = {"error": "Malformed order details response"}

        # Extract fields with fallbacks
        if isinstance(od, dict) and not od.get("error"):
            placed_at = od.get("placed_at") or od.get("created_at") or od.get("date") or entry_meta.get("placed_at") or entry_meta.get("created_at") or entry_meta.get("date")
            status_val = od.get("status") or od.get("order_status") or entry_meta.get("status") or entry_meta.get("order_status")
            total_val = od.get("total") or od.get("amount_total") or entry_meta.get("total") or entry_meta.get("amount_total")
        else:
            placed_at = entry_meta.get("placed_at") or entry_meta.get("created_at") or entry_meta.get("date")
            status_val = entry_meta.get("status") or entry_meta.get("order_status")
            total_val = entry_meta.get("total") or entry_meta.get("amount_total")

        # Status filter
        if status_norm != "any":
            if not isinstance(status_val, str) or status_val.strip().lower() != status_norm:
                continue

        # Date filters (inclusive)
        dt = _parse_dt(placed_at) if placed_at is not None else None
        if (since_dt or until_dt) and dt is None:
            # Cannot evaluate date filters without a parsable placed_at
            continue
        if since_dt and dt and dt < since_dt:
            continue
        if until_dt and dt and dt > until_dt:
            continue

        # Fetch items for summary
        item_count = 0
        items_summary_list = []
        try:
            loi = list_order_items(order_id=oid)
        except Exception as e:
            loi = {"error": str(e)}

        if isinstance(loi, str):
            try:
                loi = json.loads(loi)
            except Exception:
                loi = {"error": "Malformed order items response"}

        items_list = None
        if isinstance(loi, dict):
            if isinstance(loi.get("items"), list):
                items_list = loi.get("items")
            elif isinstance(loi.get("data"), list):
                items_list = loi.get("data")
        elif isinstance(loi, list):
            items_list = loi

        if items_list:
            for it in items_list:
                if not isinstance(it, dict):
                    continue
                qty = it.get("quantity")
                try:
                    qty = int(qty) if qty is not None else 1
                except Exception:
                    qty = 1
                item_count += qty
                name = it.get("product_name") or it.get("name") or it.get("product_type") or it.get("product") or "Item"
                opts = it.get("option_attributes") or it.get("options") or it.get("attributes")
                opts_str = ""
                if isinstance(opts, dict) and opts:
                    try:
                        keys = sorted(list(opts.keys()))
                    except Exception:
                        keys = list(opts.keys())
                    pairs = []
                    for k in keys[:3]:
                        try:
                            v = opts.get(k)
                        except Exception:
                            v = None
                        pairs.append(f"{k}: {v}")
                    opts_str = ", ".join([p for p in pairs if p])
                elif isinstance(opts, list) and opts:
                    pair_list = []
                    for elem in opts[:3]:
                        if isinstance(elem, dict) and elem:
                            k = list(elem.keys())[0]
                            pair_list.append(f"{k}: {elem.get(k)}")
                        else:
                            pair_list.append(str(elem))
                    opts_str = ", ".join(pair_list)
                label = name
                if opts_str:
                    label = f"{label} ({opts_str})"
                label = f"{label} x{qty}"
                items_summary_list.append(label)

        # Normalize total to float if possible
        total_out = total_val
        if isinstance(total_val, str):
            tv = total_val.strip().replace("$", "").replace(",", "")
            try:
                total_out = float(tv)
            except Exception:
                total_out = total_val
        elif isinstance(total_val, (int, float)):
            try:
                total_out = float(total_val)
            except Exception:
                total_out = total_val

        results.append({
            "order_id": oid,
            "placed_at": placed_at,
            "status": status_val,
            "total": total_out,
            "item_count": item_count,
            "items_summary": "; ".join(items_summary_list) if items_summary_list else ""
        })

    # Sorting
    def _sort_key(r):
        d = r.get("placed_at")
        dt = _parse_dt(d) if d is not None else None
        return dt or datetime.min

    results_sorted = sorted(results, key=_sort_key, reverse=(sort_norm == "newest"))

    # Apply limit
    if lim is not None:
        if lim == 0:
            results_sorted = []
        else:
            results_sorted = results_sorted[:lim]

    return {
        "user_id": user_id,
        "filters": {"status": status_norm, "since": since, "until": until, "limit": lim, "sort": sort_norm},
        "count": len(results_sorted),
        "orders": results_sorted
    }


@mcp.tool()
def calculate_item_price_difference(order_id, item_ids, new_item_ids):
    """
{
  "type": "function",
  "function": {
    "name": "calculate_item_price_difference",
    "description": "Calculates per-pair and total price differences between current items and proposed new items by fetching each item's current price via get_item_details. Optionally validates provided items against an order (if order_id is given). Duplicate IDs represent quantity (handled by pairwise inputs). Returns detailed pair results and totals.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "description": "Optional order id (with or without leading '#'). If provided, current items are validated against this order."
        },
        "item_ids": {
          "type": "string",
          "description": "JSON array or comma-separated list of original item_ids. Duplicates represent quantity. Must align by position with new_item_ids."
        },
        "new_item_ids": {
          "type": "string",
          "description": "JSON array or comma-separated list of proposed new item_ids. Each position aligns with item_ids. Duplicates represent quantity."
        }
      },
      "required": [
        "item_ids",
        "new_item_ids"
      ]
    }
  }
}
    """
    import json
    from collections import Counter

    # Parse helper for item id lists
    def _parse_id_list(s):
        if s is None:
            return None, "missing"
        s = s.strip()
        if not s:
            return [], None
        try:
            data = json.loads(s)
            if isinstance(data, list):
                out = []
                for v in data:
                    out.append(str(v))
                return out, None
            elif isinstance(data, str):
                s2 = data
            else:
                s2 = s
        except Exception:
            s2 = s
        parts = [p.strip() for p in s2.split(',') if p.strip()]
        return parts, None

    orig_list, e1 = _parse_id_list(item_ids)
    new_list, e2 = _parse_id_list(new_item_ids)
    if e1 == "missing" or e2 == "missing":
        return {"error": "Missing required parameter(s): item_ids and/or new_item_ids"}
    if orig_list is None or new_list is None:
        return {"error": "Failed to parse item_ids or new_item_ids"}
    if len(orig_list) == 0 or len(new_list) == 0:
        return {"error": "item_ids and new_item_ids must be non-empty"}
    if len(orig_list) != len(new_list):
        return {"error": "item_ids and new_item_ids must have the same length"}

    # Normalize order_id if provided
    normalized_order_id = None
    if order_id is not None:
        oid = order_id.strip()
        if oid:
            normalized_order_id = oid if oid.startswith('#') else f'#{oid}'

    # Optional: validate provided original items exist in the order (with quantities)
    order_validation = {}
    if normalized_order_id:
        try:
            order_info = list_order_items(normalized_order_id)
            avail_counts = Counter()
            items_field = None
            if isinstance(order_info, dict) and isinstance(order_info.get("items"), list):
                items_field = order_info.get("items")
            if isinstance(items_field, list):
                for it in items_field:
                    try:
                        iid = str(it.get("item_id")) if it.get("item_id") is not None else None
                        qty = it.get("quantity", 1)
                        try:
                            qty = int(qty)
                        except Exception:
                            qty = 1
                        if iid:
                            avail_counts[iid] += qty
                    except Exception:
                        continue
                provided_counts = Counter(orig_list)
                missing_or_exceeded = {}
                for iid, cnt in provided_counts.items():
                    if cnt > avail_counts.get(iid, 0):
                        missing_or_exceeded[iid] = {"requested": cnt, "available_in_order": avail_counts.get(iid, 0)}
                if missing_or_exceeded:
                    order_validation["mismatch"] = True
                    order_validation["details"] = missing_or_exceeded
                else:
                    order_validation["mismatch"] = False
            else:
                order_validation["note"] = "Could not validate items against order; missing items list."
        except Exception as e:
            order_validation["error"] = f"Failed to validate against order_id {normalized_order_id}: {str(e)}"

    # Fetch details for all involved item ids
    unique_ids = list(dict.fromkeys(orig_list + new_list))
    details_cache = {}

    def _extract_price(d):
        keys = ["price", "current_price", "currentPrice", "unit_price", "unitPrice"]
        for k in keys:
            if isinstance(d, dict) and k in d:
                val = d[k]
                if isinstance(val, str):
                    val = val.replace("$", "").replace(",", "").strip()
                try:
                    return float(val)
                except Exception:
                    continue
        return None

    def _extract_product_id(d):
        for k in ["product_id", "productId", "parent_product_id", "parentProductId"]:
            if isinstance(d, dict) and k in d:
                return str(d[k])
        return None

    for iid in unique_ids:
        try:
            resp = get_item_details(iid)
            if isinstance(resp, dict) and "error" in resp:
                details_cache[iid] = {"error": resp.get("error")}
            else:
                details_cache[iid] = resp
        except Exception as e:
            details_cache[iid] = {"error": str(e)}

    pairs = []
    total_diff = 0.0
    valid_pairs = 0
    invalid_pairs = 0

    for old_id, new_id in zip(orig_list, new_list):
        pair = {"original_item_id": old_id, "new_item_id": new_id}
        old_detail = details_cache.get(old_id)
        new_detail = details_cache.get(new_id)
        pair_errors = []

        # Original item
        if not isinstance(old_detail, dict) or "error" in old_detail:
            pair_errors.append(f"Failed to fetch original item {old_id}: {old_detail.get('error') if isinstance(old_detail, dict) else 'unknown error'}")
            old_price = None
            old_pid = None
        else:
            old_price = _extract_price(old_detail)
            old_pid = _extract_product_id(old_detail)
            pair["original_price"] = old_price
            pair["original_product_id"] = old_pid
            if old_pid is None:
                pair_errors.append(f"Missing product_id for original item {old_id}")
            if old_price is None:
                pair_errors.append(f"Missing price for original item {old_id}")

        # New item
        if not isinstance(new_detail, dict) or "error" in new_detail:
            pair_errors.append(f"Failed to fetch new item {new_id}: {new_detail.get('error') if isinstance(new_detail, dict) else 'unknown error'}")
            new_price = None
            new_pid = None
        else:
            new_price = _extract_price(new_detail)
            new_pid = _extract_product_id(new_detail)
            pair["new_price"] = new_price
            pair["new_product_id"] = new_pid
            if new_pid is None:
                pair_errors.append(f"Missing product_id for new item {new_id}")
            if new_price is None:
                pair_errors.append(f"Missing price for new item {new_id}")

        same_product = (old_pid is not None and new_pid is not None and old_pid == new_pid)
        pair["same_product"] = same_product
        if old_pid is not None and new_pid is not None and old_pid != new_pid:
            pair_errors.append("Original and new items are from different products; modify/exchange may not be allowed.")

        if old_price is None or new_price is None:
            pair["price_difference"] = None
            invalid_pairs += 1
        else:
            diff = round(new_price - old_price + 1e-9, 2)
            pair["price_difference"] = diff
            total_diff += diff
            valid_pairs += 1

        if pair_errors:
            pair["error"] = "; ".join(pair_errors)

        pairs.append(pair)

    result = {
        "order_id": normalized_order_id,
        "pairs": pairs,
        "total_difference": round(total_diff + 1e-9, 2),
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs
    }

    if order_validation:
        result["order_validation"] = order_validation

    increase = sum(p.get("price_difference", 0) for p in pairs if isinstance(p.get("price_difference"), (int, float)) and p.get("price_difference", 0) > 0)
    decrease = sum(-p.get("price_difference", 0) for p in pairs if isinstance(p.get("price_difference"), (int, float)) and p.get("price_difference", 0) < 0)
    result["increase_total"] = round(increase + 1e-9, 2)
    result["refund_total"] = round(decrease + 1e-9, 2)
    if result["total_difference"] > 0:
        result["amount_to_pay"] = result["total_difference"]
    elif result["total_difference"] < 0:
        result["amount_to_refund"] = -result["total_difference"]

    return result


if __name__ == "__main__":
    mcp.run(transport='stdio')
