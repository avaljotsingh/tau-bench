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
from litellm import completion as llm_completion
from azure.core.exceptions import ServiceRequestError, ServiceResponseError, ClientAuthenticationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Transient TRAPI/network failures (connection aborted, read timeout, DNS
# getaddrinfo failures) surface as ServiceRequestError / ServiceResponseError
# (ServiceResponseTimeoutError subclasses the latter). Retry these with backoff
# so a momentary blip doesn't kill a multi-hour run. HttpResponseError (4xx/5xx,
# e.g. a bad-request 400) is intentionally NOT retried.
# ClientAuthenticationError covers mid-run token expiry (~1h az/TRAPI token TTL): retrying
# re-calls credential.get_token(), minting a fresh token as long as the az session is valid,
# so a long run self-heals instead of crashing.
_RETRYABLE_TRAPI_ERRORS = (ServiceRequestError, ServiceResponseError, ClientAuthenticationError)

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

# Model/deployment are driven by the TRAPI_* env vars set by the entrypoints
# (run.py / libgen_experiment.py), defaulting to the gpt-5 deployment.
api_version = os.environ.get("TRAPI_API_VERSION", '2025-03-01-preview')  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
model_name = os.environ.get("TRAPI_MODEL_NAME", 'gpt-5')  # Ensure this is a valid model name
model_version = os.environ.get("TRAPI_MODEL_VERSION", '2024-11-20')  # Ensure this is a valid model version
deployment_name = os.environ.get("TRAPI_DEPLOYMENT_NAME", "gpt-5_2025-08-07") #re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models



# Note: Check out the other model deployments here - https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information
# api_version = '2025-03-01-preview'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
# model_name = 'gpt-4.1'  # Ensure this is a valid model name
# model_version = '2025-04-14'  # Ensure this is a valid model version
# deployment_name = "gpt-4.1_2025-04-14" #re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
instance = os.environ.get("TRAPI_INSTANCE", "redmond/interactive/openai") #'gcr/shared/openai' # See https://aka.ms/trapi/models for the instance name
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
    # Prefer native dump methods when available
    try:
        if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
            return x.model_dump()
        if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
            return x.to_dict()
        if hasattr(x, "dict") and callable(getattr(x, "dict")):
            return x.dict()
        if isinstance(x, dict):
            return x
        # Fallback for Azure SDK objects that expose `_data`
        if hasattr(x, "_data"):
            model = RecursiveModel.from_data(x._data)
            return model.model_dump()
        # Last resort: attempt to serialize __dict__-like objects
        model = RecursiveModel.from_data(getattr(x, "__dict__", x))
        return model.model_dump()
    except Exception:
        # Safe fallback
        return x


def completion(*args, **kwargs):
    """
    Dispatches chat completion calls:
    - OpenAI-compatible (e.g., vLLM) via LiteLLM when provider/base_url indicate OpenAI API
    - Azure TRAPI client otherwise (existing path)
    """
    provider = kwargs.get("custom_llm_provider")
    base_url = kwargs.pop("base_url", None) or os.environ.get("OPENAI_API_BASE") or os.environ.get("VLLM_BASE_URL")

    use_openai_compatible = (
        (provider in ("openai", "openai_compatible")) or
        (base_url is not None)
    )
    use_openai_compatible = False

    if use_openai_compatible:
        # Ensure tool calling occurs when tools are provided
        tools = kwargs.get("tools")
        # Only enable auto tool choice if explicitly requested and supported by server flags
        enable_auto = os.environ.get("TAU_ENABLE_AUTO_TOOL_CHOICE") == "1" or os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE") == "1"
        if tools is not None and "tool_choice" not in kwargs and enable_auto:
            kwargs["tool_choice"] = "auto"
        # Honor base_url/api_key if provided via env/kwargs
        if base_url is not None:
            kwargs["base_url"] = base_url
        api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
        if api_key is not None:
            kwargs["api_key"] = api_key
        start_time = time.time()
        res = llm_completion(*args, **kwargs)
        end_time = time.time()
        llm_time.record_time(end_time - start_time)
        return res

    # Default Azure TRAPI path (existing behavior)
    sig = inspect.signature(client.complete)
    allowed_params = sig.parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
    # gpt-5 / reasoning deployments (o1, o3, gpt-5*) only accept the default
    # temperature (1). The tau-bench agents pass 0.1/0.2, which 400s; drop the
    # param for these models so the deployment default is used.
    _reasoning_model = any(tok in deployment_name.lower() for tok in ("gpt-5", "o1", "o3", "o4"))
    if _reasoning_model and "temperature" in filtered_kwargs:
        filtered_kwargs.pop("temperature", None)

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(_RETRYABLE_TRAPI_ERRORS),
    )
    def _complete_with_retry():
        return client.complete(*args, **filtered_kwargs)

    start_time = time.time()
    res = _complete_with_retry()
    end_time = time.time()
    llm_time.record_time(end_time - start_time)
    return res


# ---------------------------------------------------------------------------
# Separate STRONG-model client for tool GENERATION, decoupled from the agent.
# The agent's deployment is env-driven and may be a WEAK model (to create
# failures + headroom). Tools should still be designed by a strong model, so
# lib_gen routes its calls here. Defaults to gpt-5; override via LIBGEN_GEN_*.
# ---------------------------------------------------------------------------
gen_deployment_name = os.environ.get("LIBGEN_GEN_DEPLOYMENT", "gpt-5_2025-08-07")
gen_instance = os.environ.get("LIBGEN_GEN_INSTANCE", instance)
gen_api_version = os.environ.get("LIBGEN_GEN_API_VERSION", api_version)
gen_endpoint = f'https://trapi.research.microsoft.com/{gen_instance}/deployments/' + gen_deployment_name
gen_client = ChatCompletionsClient(
    endpoint=gen_endpoint, credential=credential, credential_scopes=scopes, api_version=gen_api_version,
)


def gen_completion(*args, **kwargs):
    """Chat completion via the STRONG generation model (separate from the agent model)."""
    sig = inspect.signature(gen_client.complete)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if any(tok in gen_deployment_name.lower() for tok in ("gpt-5", "o1", "o3", "o4")):
        filtered.pop("temperature", None)

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(_RETRYABLE_TRAPI_ERRORS),
    )
    def _c():
        return gen_client.complete(*args, **filtered)

    return _c()
