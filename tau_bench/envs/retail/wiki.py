# Copyright Sierra

import os
# from tau_bench.envs.retail.get_tools import get_function_infos

# res = get_function_infos()
# print(res)
# kldfj

FOLDER_PATH = os.path.dirname(__file__)

with open(os.path.join(FOLDER_PATH, "wiki_one_shot.md"), "r") as f:
    WIKI = f.read()
