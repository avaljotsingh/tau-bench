import os
import openai ###
import azure.identity ###

from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens
from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential

DEFAULT_OPENAI_MODEL = "gpt-4o-2024-08-06"
API_KEY_ENV_VAR = "OPENAI_API_KEY"

PRICE_PER_INPUT_TOKEN_MAP = {
    "gpt-4o-2024-08-06": 2.5 / 1000000,
    "gpt-4o": 5 / 1000000,
    "gpt-4o-2024-08-06": 2.5 / 1000000,
    "gpt-4o-2024-05-13": 5 / 1000000,
    "gpt-4-turbo": 10 / 1000000,
    "gpt-4-turbo-2024-04-09": 10 / 1000000,
    "gpt-4": 30 / 1000000,
    "gpt-4o-mini": 0.15 / 1000000,
    "gpt-4o-mini-2024-07-18": 0.15 / 1000000,
    "gpt-3.5-turbo": 0.5 / 1000000,
    "gpt-3.5-turbo-0125": 0.5 / 1000000,
    "gpt-3.5-turbo-instruct": 1.5 / 1000000,
}
INPUT_PRICE_PER_TOKEN_FALLBACK = 10 / 1000000

CAPABILITY_SCORE_MAP = {
    "gpt-4o-2024-08-06": 0.8,
    "gpt-4o": 0.8,
    "gpt-4o-2024-08-06": 0.8,
    "gpt-4o-2024-05-13": 0.8,
    "gpt-4-turbo": 0.9,
    "gpt-4-turbo-2024-04-09": 0.9,
    "gpt-4": 0.8,
    "gpt-4o-mini": 0.5,
    "gpt-4o-mini-2024-07-18": 0.5,
    "gpt-3.5-turbo": 0.3,
    "gpt-3.5-turbo-0125": 0.3,
}
CAPABILITY_SCORE_FALLBACK = 0.3

# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {}
# TODO: implement
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 0.0

MAX_CONTEXT_LENGTH_MAP = {
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4": 8192,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-0125": 16385,
}
MAX_CONTEXT_LENGTH_FALLBACK = 128000


class OpenAIModel(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        from openai import AsyncOpenAI, OpenAI

        if model is None:
            self.model = DEFAULT_OPENAI_MODEL
        else:
            self.model = model

        #if api_key is None:
        #    api_key = os.getenv(API_KEY_ENV_VAR)
        #    if api_key is None:
        #        raise ValueError(f"{API_KEY_ENV_VAR} environment variable is not set")
        #self.client = OpenAI(api_key=api_key)
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

    # Note: Check out the other model deployments here - https://dev.azure.com/msresearch/TRAPI/_wiki/wikis/TRAPI.wiki/15124/Deployment-Model-Information
        api_version = '2025-03-01-preview'  # Ensure this is a valid API version see: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
        deployment_name = "gpt-4o_2024-11-20" #re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')  # If your Endpoint doesn't have harmonized deployment names, you can use the deployment name directly: see: https://aka.ms/trapi/models
        instance = "redmond/interactive/openai" #'gcr/shared/openai' # See https://aka.ms/trapi/models for the instance name
        endpoint = f'https://trapi.research.microsoft.com/{instance}/deployments/'+deployment_name

        self.client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=credential,
        credential_scopes=scopes,
        api_version=api_version
        )
        #azure_ad_token_provider=azure.identity.get_bearer_token_provider(azure.identity.AzureCliCredential(), "https://cognitiveservices.azure.com/.default"))
        #self.async_client = AsyncOpenAI(api_key=api_key)
        self.temperature = temperature

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
        msgs = self.build_generate_message_state(messages)
        res = self.client.complete(
            model=self.model,
            messages=msgs,
            temperature=wrap_temperature(temperature),
            #response_format={"type": "json_object" if force_json else "text"},
        )
        return self.handle_generate_message_response(
            prompt=msgs, content=res.choices[0].message.content, force_json=force_json
        )

    def get_approx_cost(self, dp: Datapoint) -> float:
        cost_per_token = PRICE_PER_INPUT_TOKEN_MAP.get(self.model, INPUT_PRICE_PER_TOKEN_FALLBACK)
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=cost_per_token)

    def get_latency(self, dp: Datapoint) -> float:
        latency_per_output_token = LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        )
        return approx_cost_for_datapoint(dp=dp, price_per_input_token=latency_per_output_token)

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def supports_dp(self, dp: Datapoint) -> bool:
        prompt = approx_prompt_str(dp)
        return approx_num_tokens(prompt) <= MAX_CONTEXT_LENGTH_MAP.get(
            self.model, MAX_CONTEXT_LENGTH_FALLBACK
        )
 