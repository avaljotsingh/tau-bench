from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional, Union

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME
from tau_bench.agents.utils import *

class Role(str, Enum):
    USER = "USER"
    ENV = "ENV"
    ACTION_AGENT = "ACTION_AGENT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_OUTPUT = "TOOL_OUTPUT"
    PRECONDITION_AGENT = "PRECONDITION_AGENT"
    PRECONDITION_OUTPUT = "PRECONDITION_OUTPUT"
    POSTCONDITION_AGENT = "POSTCONDITION_AGENT"
    POSTCONDITION_OUTPUT = "POSTCONDITION_OUTPUT"


class User(BaseModel):
    first_name: str = "-"
    last_name: str = "-"
    zip: str = "-"
    user_id: str = "-"
    email: str = "-"

class Item(BaseModel):
    item_id: str = "-"
    properties: Dict[str, Any] = {}
    availability: str = "-"
    price: str = '-'

class Product(BaseModel):
    product_id: str = "-"
    product_name: str = "-"
    items: List[Item] = []

class Order(BaseModel):
    order_id: str = "-"
    items: List[Item] = []
    status: str = "-"

class TaskType(str, Enum):
    ValidateUser = "validate_user"
    FindUserByEmail = "find_user_id_by_email"
    FindUserByZip = "find_user_id_by_name_zip"
    GetUserDetails = "get_user_details"
    GetProductDetails = "get_product_details"
    GetOrderDetails = "get_order_details"
    GetUserInput = "get_input_from_user"
    ListAllProducts = "list_all_product_types"
    Calculate = "calculate"
    Think = "think"
    TransferToHumanAgent = "transfer_to_human_agent"
    CancelPendingOrder = "cancel_pending_order"
    ModifyPendingOrderAddress = "modify_pending_order_address"
    ModifyPendingOrderItems = "modify_pending_order_items"
    ModifyPendingOrderPayment = "modify_pending_order_payment"
    ModifyUserAddress = "modify_user_address"
    ReturnDeliveredOrderItems = "return_delivered_order_items"
    ExchangeDeliveredOrderItems = "exchange_delivered_order_items"

def get_func_from_tasktype(tasktype):
    if tasktype == TaskType.ValidateUser:
        return ['find_user_id_by_email', 'find_user_id_by_name_zip']
    elif tasktype == TaskType.FindUserByEmail:
        return ['find_user_id_by_email']
    elif tasktype == TaskType.FindUserByZip:
        return ['find_user_id_by_name_zip']
    elif tasktype == TaskType.GetUserDetails:
        return ['get_user_details']
    elif tasktype == TaskType.GetProductDetails:
        return ['get_product_details']
    elif tasktype == TaskType.GetOrderDetails:
        return ['get_order_details']
    elif tasktype == TaskType.GetUserInput:
        return ['get_input_from_user']
    elif tasktype == TaskType.ListAllProducts:
        return ['list_all_product_types']
    elif tasktype == TaskType.Calculate:
        return ['calculate']
    elif tasktype == TaskType.Think:
        return ['think']
    elif tasktype == TaskType.TransferToHumanAgent:
        return ['transfer_to_human_agent']
    elif tasktype == TaskType.CancelPendingOrder:
        return ['cancel_pending_order']
    elif tasktype == TaskType.ModifyPendingOrderAddress:
        return ['modify_pending_order_address']
    elif tasktype == TaskType.ModifyPendingOrderItems:
        return ['modify_pending_order_items']
    elif tasktype == TaskType.ModifyPendingOrderPayment:
        return ['modify_pending_order_payment']
    elif tasktype == TaskType.ModifyUserAddress:
        return ['modify_user_address']
    elif tasktype == TaskType.ReturnDeliveredOrderItems:
        return ['return_delivered_order_items']
    elif tasktype == TaskType.ExchangeDeliveredOrderItems:
        return ['exchange_delivered_order_items']
    else:
        print(tasktype)
        raise Exception("Unknown tasktype")

class Task:
    def __init__(self, tasktype, args={}):
        self.tasktype = tasktype
        self.possible_funcs = get_func_from_tasktype(tasktype)
        self.args = args

    def get_description(self):
        return f'{self.tasktype} using the functions {self.possible_funcs} with arguiments ({self.args})'


class TaskGraph:
    def __init__(self, tasks=[], dependencies={}):
        self.nodes = tasks
        self.edges = dependencies
    
    def find_roots(self):
        return [node for node in self.nodes if len(self.edges[node]) == 0]
    
    def add_task(self, task: Task):
        self.nodes.append(task)
        if task not in self.edges:
            self.edges[task] = []

    def add_edge(self, task1: Task, task2: Task):
        self.edges[task1].append(task2)
        # self.edges.append((task1, task2))
    
