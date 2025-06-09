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
user_id = find_user_id_by_name_zip(name="Mei Kovacs", zip_code="28236")
orders = get_order_details(user_id=user_id)
water_bottle_order = None
desk_lamp_order = None
for order in orders:
    for item in order['items']:
        if "water bottle" in item['product_name'].lower():
            water_bottle_order = {"order_id": order['order_id'], "item_id": item['item_id']}
        if "desk lamp" in item['product_name'].lower():
            desk_lamp_order = {"order_id": order['order_id'], "item_id": item['item_id']}
    if water_bottle_order and desk_lamp_order:
        break
if not water_bottle_order or not desk_lamp_order:
    transfer_to_human_agents(reason="Could not locate one or both items in the user's orders.")
desk_lamp_details = get_product_details(product_name="desk lamp")
suitable_lamps = []
for lamp in desk_lamp_details:
    if "less bright" in lamp['description'].lower():
        if "battery-powered" in lamp['power_source'].lower():
            suitable_lamps.append(lamp)
        elif "usb" in lamp['power_source'].lower():
            suitable_lamps.append(lamp)
        elif "ac" in lamp['power_source'].lower():
            suitable_lamps.append(lamp)
if not suitable_lamps:
    transfer_to_human_agents(reason="Could not find a suitable replacement for the desk lamp.")
replacement_lamp = suitable_lamps[0]  # Choose the first suitable lamp as default
get_input_from_user(prompt=f"Do you want to exchange the desk lamp for {replacement_lamp['product_name']}?")
if user_input == "yes":
    exchange_delivered_order_items(
        user_id=user_id,
        exchanges=[
            {"order_id": water_bottle_order['order_id'], "item_id": water_bottle_order['item_id'], "replacement": "bigger size"},
            {"order_id": desk_lamp_order['order_id'], "item_id": desk_lamp_order['item_id'], "replacement": replacement_lamp['product_id']}
        ]
    )
else:
    transfer_to_human_agents(reason="User did not confirm the replacement for the desk lamp.")