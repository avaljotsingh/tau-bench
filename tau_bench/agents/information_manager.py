import json
from termcolor import colored

from tau_bench.trapi_infer import completion
from tau_bench.agents.state import *

class InformationManager:
    def __init__(self, information):
        self.information = information

    def clean_code_block(self, code_str: str) -> str:
        new_lines = []
        code_started = False
        lines = code_str.strip().splitlines()
        for line in lines:
            if code_started and ("```" in line):
                break
            if code_started:
                new_lines.append(line)
            if ("```python" in line):
                code_started = True
        # if lines[0].strip().startswith("```"):
        #     code_started = True
        #     lines = lines[1:]
        # if lines and lines[-1].strip().startswith("```"):
        #     lines = lines[:-1]
        return "\n".join(new_lines)

    def add_new_knowledge(self, new_knowledge, allowed_classes):
        """
        Executes a knowledge string and updates the `information` dictionary
        with any created variables that are instances of the allowed classes.
        
        Parameters:
            information: Dictionary to update.
            knowledge_str: A string of code that defines new knowledge (e.g., user_1 = User(...)).
            allowed_classes: A dictionary of allowed class names to class types (e.g., {"User": User, ...}).
        """
        # Limit scope to only allowed classes
        # exec_scope = allowed_classes.copy()
        new_knowledge = self.clean_code_block(new_knowledge)
        exec_scope = allowed_classes.copy()
        local_vars = {}

        exec(new_knowledge, exec_scope, local_vars)

        # Add only allowed objects to information
        for name, obj in local_vars.items():
            if any(isinstance(obj, cls) for cls in allowed_classes.values()):
                self.information[name] = obj

    def parse_user_message(self, message):

        if len(list(self.information.keys())) == 0:
            information = ""
        else:
            print(self.information)
            information = str(self.information)
            # information = json.dumps(self.information)
        SYSTEM_PROMPT = f'''
I have the following information currently: {information}. 
The user will provide further information: {message}. 
Please update the information accordingly. 
The format shopuld be the following: 
information is a dictonary with the keys as objects of the following classes: 
You can use the following class objects: 
class User(BaseModel):
    first_name: str = "-"
    last_name: str = "-"
    zip: str = "-"
    user_id: str = "-"
    email: str = "-"

class Item(BaseModel):
    item_id: str = "-"
    properties: Dict[str, Any] = {{}}
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

You should name the variables as user_1, order_1, etc.
Do not define the classes again. Just define the objects or update the information in the existing objects.
'''
        context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}]
        res = completion(messages=context)
        new_information = res.choices[0].message.content
        # print(new_information)
        # self.update_information(new_information)
        return new_information

    def update_information(self, user_message):
        print(colored('###########Information manager activated###########', 'red'))
        new_knowledge = self.parse_user_message(user_message)
        print(colored(new_knowledge, 'yellow'))
        self.add_new_knowledge(new_knowledge, allowed_classes={"User": User, "Item": Item, "Product": Product, "Order": Order})
        print(colored(self.information, 'blue'))
        print(colored('###########Information manager deactivated###########', 'red'))
    