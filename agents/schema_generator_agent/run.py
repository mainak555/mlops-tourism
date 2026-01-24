
from agents.agent_util import create_kernel, load_schema, validate_schema
from pandas.api.types import is_integer_dtype, is_float_dtype
from semantic_kernel.functions import KernelArguments
import pandas as pd
import json
import os

async def run_schema_generator(df: pd.DataFrame, input_features: list) -> dict:
    """
    1. Extracts 'Ground Truth' from DF.
    2. Invokes SK Plugin.
    3. Validates against config.json schema.
    """

    kernel = create_kernel()
    schema_generator = kernel.add_plugin(
        plugin_name="schema_generator_plugin",
        parent_directory="agents/plugins"
    )

    cat_inputs, num_inputs, text_inputs = {}, {}, {}
    for col in sorted(input_features):
        series = df[col].dropna()
        unique_values = sorted(series.unique().tolist())
        unique_count = len(unique_values)

        if series.dtype == "object":
            if unique_count <= 10:
                cat_inputs[col] = {
                    "input_type": "select",
                    "options": {str(v): str(v) for v in unique_values}
                }
            else:
                text_inputs[col] = {
                    "input_type": "text"
                }
        elif is_integer_dtype(series):
            if unique_count <= 5:
                cat_inputs[col] = {
                    "input_type": "select",
                    "options": [int(v) for v in unique_values]
                }
            else:
                num_inputs[col] = {
                    "input_type": "number",
                    "min": int(series.min()),
                    "max": int(series.max()),
                    "is_int": True
                }
        elif is_float_dtype(series):
            num_inputs[col] = {
                "input_type": "number",
                "min": float(series.min()),
                "max": float(series.max()),
                "is_int": False
            }
        else:
            text_inputs[col] = {
                "input_type": "text"
            }

    args = KernelArguments(
        categorical_features=json.dumps(cat_inputs),
        numerical_features=json.dumps(num_inputs),
        text_features=json.dumps(text_inputs)
    )    

    print(f"number field: {num_inputs}")
    print(f"select field: {cat_inputs}")
    print(f"text field: {text_inputs}")

    result = await kernel.invoke(
        schema_generator["generate_schema"],
        arguments=args
    )

    try:
        output = json.loads(str(result))
    except json.JSONDecodeError:
        print(f"Raw Output: {result}")
        raise RuntimeError("Schema Generator Agent returned invalid JSON")

    SCHEMA_PATH = "agents/plugins/schema_generator_plugin/generate_schema/config.json"
    schema = load_schema(SCHEMA_PATH)
    validate_schema(output, schema)

    return output
