
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIChatCompletion
from jsonschema import validate, ValidationError
from semantic_kernel import Kernel
from pprint import pprint
import json
import os

def create_kernel() -> Kernel:
    kernel = Kernel()

    if os.getenv("USE_OPENAI") == "true":
        kernel.add_service(
            OpenAIChatCompletion(
                ai_model_id=os.getenv("OPENAI_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                service_id="default",
            )
        )
    else:
        kernel.add_service(AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            service_id="default",
        )) 

    return kernel

def validate_schema(output: dict, schema: dict):
    """
    Validates agent output against a JSON schema.
    Raises RuntimeError on failure.
    """
    try:
        validate(instance=output, schema=schema)
    except ValidationError as e:
        pprint(output)
        raise RuntimeError(f"Agent output schema violation: {e.message}")

def load_schema(schema_path: str) -> dict:
    with open(schema_path, "r") as f:
        cfg = json.load(f)
    return cfg["schema"]
