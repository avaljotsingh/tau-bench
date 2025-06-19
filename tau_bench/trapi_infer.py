from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential
import os
import re
import inspect
from pydantic import RootModel
from typing import Any, Dict, List, Union
import json

import time
from tau_bench.globals import *

credential = ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        # Exclude other credentials we are not interested in.
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
        # Azure ML Compute jobs that has the client id of the
        # user-assigned managed identity in it.
        # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
        # In case it is not set the ManagedIdentityCredential will
        # default to using the system-assigned managed identity, if any.
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    )
)
scopes = ["api://trapi/.default"]

# # Note: Check out the other model deployments here - https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information
# api_version = '2025-03-01-preview'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
# model_name = 'o3'  # Ensure this is a valid model name
# model_version = '2025-04-16'  # Ensure this is a valid model version
# deployment_name = "o3_2025-04-16" #re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
# instance = "redmond/interactive/openai" #'gcr/shared/openai' # See https://aka.ms/trapi/models for the instance name
# endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/'+deployment_name


# Note: Check out the other model deployments here - https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information
api_version = '2025-03-01-preview'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
model_name = 'gpt-4o'  # Ensure this is a valid model name
model_version = '2024-11-20'  # Ensure this is a valid model version
deployment_name = "gpt-4o_2024-11-20" #re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
instance = "redmond/interactive/openai" #'gcr/shared/openai' # See https://aka.ms/trapi/models for the instance name
endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/'+deployment_name

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=credential,
    credential_scopes=scopes,
    api_version=api_version
)

completion = client.complete
# response = client.complete(
#     model=model_name,
#     messages=[
#         {
#             "role": "user",
#             "content": "Give a one word answer, what is the capital of France?",
#         },
#     ]
# )
# response_content = response.choices[0].message.content
# print(response_content)


Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

class RecursiveModel(RootModel[Json]):
    @classmethod
    def from_data(cls, data):
        if isinstance(data, dict):
            # recursively wrap each value in RecursiveModel
            return cls({k: cls.from_data(v) for k, v in data.items()})
        elif isinstance(data, list):
            return cls([cls.from_data(i) for i in data])
        else:
            return cls(data)

    def model_dump(self):
        if isinstance(self.root, dict):
            return {k: v.model_dump() if isinstance(v, RecursiveModel) else v for k, v in self.root.items()}
        elif isinstance(self.root, list):
            return [v.model_dump() if isinstance(v, RecursiveModel) else v for v in self.root]
        else:
            return self.root
def model_dump(x):
    model = RecursiveModel.from_data(x._data)
    dumped = model.model_dump()
    return dumped


def completion(*args, **kwargs):
    sig = inspect.signature(client.complete)
    allowed_params = sig.parameters

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
    # contextLength.get_lengths_from_messages(filtered_kwargs['messages'])
    start_time = time.time()
    res = client.complete(*args, **filtered_kwargs)
    end_time = time.time()
    llm_time.record_time(end_time - start_time)
    return res
