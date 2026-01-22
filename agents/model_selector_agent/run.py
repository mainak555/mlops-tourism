
from semantic_kernel.functions import KernelArguments
from agents.agent_util import create_kernel
import json

async def run_model_selector(payload: dict) -> dict:
    kernel = create_kernel()

    model_selector = kernel.add_plugin(
        plugin_name="model_selector_agent",
        parent_directory="agents"
    )

    arguments = KernelArguments(input=json.dumps(payload, indent=2))

    result = await kernel.invoke(
        model_selector["select_model"],
        arguments=arguments
    )

    try:
        return json.loads(str(result))
    except json.JSONDecodeError:
        print(f"Raw Output: {result}")
        raise RuntimeError("Model Selector Agent returned invalid JSON")
