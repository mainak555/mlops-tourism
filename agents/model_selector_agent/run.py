
from agents.agent_util import create_kernel, load_schema, validate_schema
from semantic_kernel.functions import KernelArguments
import json

async def run_model_selector(payload: dict) -> dict:
    kernel = create_kernel()

    model_selector = kernel.add_plugin(
        plugin_name="model_selector_agent",
        parent_directory="agents"
    )

    args = KernelArguments(input=json.dumps(payload, indent=2))

    result = await kernel.invoke(
        model_selector["select_model"],
        arguments=args
    )

    try:
        output = json.loads(str(result))
    except json.JSONDecodeError:
        print(f"Raw Output: {result}")
        raise RuntimeError("Model Selector Agent returned invalid JSON")

    ## output validation ##
    SCHEMA_PATH = "agents/model_selector_agent/select_model/config.json"
    schema = load_schema(SCHEMA_PATH)
    validate_schema(output, schema)
    return output
