import importlib.util
import os
import inspect
from tau_bench.envs.tool import Tool  # Ensure Tool is importable

def load_tool_subclass_from_file(filepath):
    module_name = os.path.splitext(os.path.basename(filepath))[0]

    # Dynamically load the module
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the subclass of Tool in the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Tool) and obj is not Tool:
            return obj  # Return the subclass of Tool

    raise ValueError(f"No Tool subclass found in {filepath}")

def get_function_infos():

    FOLDER_PATH = os.path.dirname(__file__)
    path = os.path.join(FOLDER_PATH, 'tools')

    files = os.listdir(path)
    files = files[:-2]
    files = [os.path.join(FOLDER_PATH, 'tools', file) for file in files]

    infos = {}

    for file_path in files:
        cls = load_tool_subclass_from_file(file_path)
        instance = cls()
        info = instance.get_info()
        parameters = info['function']['parameters']['required']
        info = {'name': info['function']['name'], 'parameters': parameters}
        infos[file_path.split('\\')[-1]] = info
        

    return infos

print(get_function_infos())