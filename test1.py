from tau_bench.envs.retail.tools.calculate import calculate
from tau_bench.envs.retail.tools.cancel_pending_order import cancel_pending_order
from tau_bench.envs.retail.tools.exchange_delivered_order_items import exchange_delivered_order_items
from tau_bench.envs.retail.tools.find_user_id_by_email import find_user_id_by_email
from tau_bench.envs.retail.tools.find_user_id_by_name_zip import find_user_id_by_name_zip
from tau_bench.envs.retail.tools.get_order_details import get_order_details
from tau_bench.envs.retail.tools.get_product_details import get_product_details
from tau_bench.envs.retail.tools.get_user_details import get_user_details
from tau_bench.envs.retail.tools.list_all_product_types import list_all_product_types
from tau_bench.envs.retail.tools.modify_pending_order_address import modify_pending_order_address
from tau_bench.envs.retail.tools.modify_pending_order_items import modify_pending_order_items
from tau_bench.envs.retail.tools.modify_pending_order_payment import modify_pending_order_payment
from tau_bench.envs.retail.tools.modify_user_address import modify_user_address
from tau_bench.envs.retail.tools.return_delivered_order_items import return_delivered_order_items
from tau_bench.envs.retail.tools.think import think
from tau_bench.envs.retail.tools.transfer_to_human_agents import transfer_to_human_agents
from tau_bench.envs.retail.tools.get_input_from_user import get_input_from_user
# Step-0: gather utilities that will be needed later
# -----------------------------------------------------------
# 1) Locate the customer account
user_id = find_user_id_by_name_zip(
    first_name="Yusuf",
    last_name="Rossi",
    zip="19122"
)
assert isinstance(user_id, str) and user_id.strip(), "Failed to locate the user ID."
# -----------------------------------------------------------
# 2) Pull the order so we can identify the two items to swap
order = get_order_details(order_id="#W2378156")
assert isinstance(order, dict) and order.get("order_id") == "#W2378156", "Failed to fetch order details."
# Make sure this really is the customer’s order
assert order["user_id"] == user_id, "Order does not belong to the located user."
# -----------------------------------------------------------
# 3) Identify the item_ids for the two products mentioned
keyboard_item_id     = None
thermostat_item_id   = None
for item in order["items"]:
    name_lower = item["name"].lower()
    if "keyboard" in name_lower and keyboard_item_id is None:
        keyboard_item_id = item["item_id"]
        keyboard_price   = item["price"]
    if "thermostat" in name_lower and thermostat_item_id is None:
        thermostat_item_id = item["item_id"]
        thermostat_price   = item["price"]
assert keyboard_item_id and thermostat_item_id, "Could not locate both products inside the order."
# -----------------------------------------------------------
# 4) Look up all product types so we can fetch the catalog for each
product_types = list_all_product_types()          # returns list of dicts with "name", "product_id"
assert isinstance(product_types, list) and all(isinstance(pt, dict) for pt in product_types), "Failed to fetch product types."
def find_product_id(name_keyword: str):
    for p in product_types:
        if name_keyword.lower() in p["name"].lower():
            return p["product_id"]
    raise ValueError(f"Product type containing '{name_keyword}' not found.")
kbd_product_id   = find_product_id("keyboard")
therm_product_id = find_product_id("thermostat")
assert kbd_product_id and therm_product_id, "Failed to identify product IDs for keyboard or thermostat."
# -----------------------------------------------------------
# 5) Find suitable replacement SKUs inside each catalogue
def choose_keyboard_replacement():
    details = get_product_details(product_id=kbd_product_id)
    assert isinstance(details, dict) and "items" in details, "Failed to fetch product details for keyboards."
    rgb_clicky_full_size   = []
    clicky_full_size_only  = []
    for item in details["items"]:
        opts = item["options"]
        size_ok     = opts.get("size") == "full-size"
        switch_ok   = opts.get("switch") == "clicky"
        rgb_ok      = opts.get("backlight") == "RGB"
        if size_ok and switch_ok and rgb_ok:
            rgb_clicky_full_size.append(item)
        elif size_ok and switch_ok:
            clicky_full_size_only.append(item)
    assert rgb_clicky_full_size or clicky_full_size_only, "No suitable keyboard replacements found."
    if rgb_clicky_full_size:
        # Prefer same or lower price; else the cheapest among them
        best = sorted(rgb_clicky_full_size, key=lambda x: x["price"])[0]
    else:
        best = sorted(clicky_full_size_only, key=lambda x: x["price"])[0]
    return {
        "item_id": best["item_id"],
        "price":   best["price"],
        "description": best["name"]
    }
def choose_thermostat_replacement():
    details = get_product_details(product_id=therm_product_id)
    assert isinstance(details, dict) and "items" in details, "Failed to fetch product details for thermostats."
    google_compatible = []
    for item in details["items"]:
        opts = item["options"]
        if "google" in opts.get("works_with", "").lower():
            google_compatible.append(item)
    assert google_compatible, "No Google-Home compatible thermostat available."
    # Choose the closest price match (smallest absolute delta)
    best = sorted(google_compatible, key=lambda x: abs(x["price"] - thermostat_price))[0]
    return {
        "item_id": best["item_id"],
        "price":   best["price"],
        "description": best["name"]
    }
keyboard_repl   = choose_keyboard_replacement()
thermostat_repl = choose_thermostat_replacement()
assert isinstance(keyboard_repl, dict) and isinstance(thermostat_repl, dict), "Failed to determine replacements."
# -----------------------------------------------------------
# 6) Compute price deltas & summarise
delta_keyboard   = keyboard_repl["price"]   - keyboard_price
delta_thermostat = thermostat_repl["price"] - thermostat_price
total_delta      = delta_keyboard + delta_thermostat
summary_lines = [
    "Proposed replacements:",
    f"• Mechanical keyboard  →  {keyboard_repl['description']} (item-id {keyboard_repl['item_id']})",
    f"     price change: {'+' if delta_keyboard>0 else ''}{delta_keyboard:.2f} USD",
    f"• Smart thermostat     →  {thermostat_repl['description']} (item-id {thermostat_repl['item_id']})",
    f"     price change: {'+' if delta_thermostat>0 else ''}{delta_thermostat:.2f} USD",
    "",
    f"Net price difference for the two items combined: {'+' if total_delta>0 else ''}{total_delta:.2f} USD.",
    "There are no restocking fees for standard exchanges.",
    "",
    "Exchange process:",
    "• As soon as you confirm, we’ll email a prepaid return label.",
    "• Please drop off the original items within 14 days.",
    "• Replacements are shipped once the carrier scans your return (usually within 1–2 business days).",
    "• Any extra charge or refund will be applied to the original payment method automatically.",
]
print("\n".join(summary_lines))
# -----------------------------------------------------------
# 7) Ask for the customer’s approval (yes/no).
confirmation = get_input_from_user(
    thought="Would you like to proceed with the above exchange? (yes / no)"
)
assert confirmation is not None, "Failed to fetch user confirmation."
if str(confirmation).strip().lower().startswith("y"):
    # We will use the first payment method that was charged on the order
    payment_method_id = order["payment_history"][0]["payment_method_id"]
    result = exchange_delivered_order_items(
        order_id     = "#W2378156",
        item_ids     = [keyboard_item_id, thermostat_item_id],
        new_item_ids = [keyboard_repl["item_id"], thermostat_repl["item_id"]],
        payment_method_id = payment_method_id
    )
    assert isinstance(result, dict), "Exchange process failed."
    print("Exchange request placed successfully. Here is a summary:")
    print(result)
else:
    print("No changes have been made to your order. Let us know anytime if you’d like to revisit the exchange.")