from typing import List, Type

import opengradient as og
import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests
from pydantic import BaseModel, Field

from langchain_opengradient.toolkits import OpenGradientToolkit
from langchain_opengradient.tools import OpenGradientTool

"""
    # Example-ish
        >>> class ClassifierInput(BaseModel):
        ...     query: str = Field(description="User query to analyze")
        ...     parameters: dict = Field(description="Additional parameters")
        >>> def get_input():
        ...     return {"text": "Sample input text"}
        >>> def format_output(output):
        ...     return str(output.get("class", "Unknown"))
        >>> # Create a LangChain tool
        >>> langchain_tool = create_og_model_tool(
        ...     tool_type=ToolType.LANGCHAIN,
        ...     model_cid="Qm...",
        ...     tool_name="text_classifier",
        ...     input_getter=get_input,
        ...     output_formatter=format_output,
        ...     input_schema=ClassifierInput
        ...     tool_description="Classifies text into categories"
    

    class ExampleInputSchema(BaseModel):
        open_high_low_close: List[List] = Field(description="[Open, High, Low, Close] prices for the 10 most recent")

    def get_input():
        return {"text": "Sample input text"}

    def format_output(output):
        return str(output.get("class", "Unknown"))
    
    toolkit = OpenGradientToolkit()

    toolkit.create_run_model_tool(
        model_cid="",
        tool_name="",
        input_getter="",
        output_formatter="",
        input_schema="",
        tool_description="",
        inference_mode="",
    )

    mock_tool = MagicMock()
    mock_tool.tool_type = ToolType.LANGCHAIN
    mock_tool.model_cid = ""
    
    with patch("opengradient.create_run_model_tool") as mock_create_run_model_tool:
        mock_create_run_model_tool.assert_called_once_with(
            model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
            tool_name="one_hour_eth_usdt_volatility",
            input_getter=get_input,
            output_formatter=format_output,
            input_schema=ExampleInputSchema,
            tool_description="Gets the live 1 hour volatility measurement for the ETH/USDT trading pair.",
            inference_mode=og.InferenceMode.VANILLA,
        )
"""


class InputSchema(BaseModel):
    open_high_low_close: List[List] = Field(
        description="[Open, High, Low, Close] prices for the 10 most recent"
    )


def input_getter():
    return {
        "open_high_low_close": [
            [2535.79, 2535.79, 2505.37, 2515.36],
            [2515.37, 2516.37, 2497.27, 2506.94],
            [2506.94, 2515, 2506.35, 2508.77],
            [2508.77, 2519, 2507.55, 2518.79],
            [2518.79, 2522.1, 2513.79, 2517.92],
            [2517.92, 2521.4, 2514.65, 2518.13],
            [2518.13, 2525.4, 2517.2, 2522.6],
            [2522.59, 2528.81, 2519.49, 2526.12],
            [2526.12, 2530, 2524.11, 2529.99],
            [2529.99, 2530.66, 2525.29, 2526],
        ]
    }


def output_formatter(output):
    result = format(float(output["Y"]), ".10%")
    return f"One hour ETH/USDT volatility prediction: {result}%"


@pytest.mark.usefixtures("mock_env")
class TestOpenGradientRunModelToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[BaseTool]:
        toolkit = OpenGradientToolkit()
        tool = toolkit.create_run_model_tool(
            model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
            tool_name="one_hour_eth_usdt_volatility",
            input_getter=input_getter,
            output_formatter=output_formatter,
            input_schema=InputSchema,
            tool_description="Generate the live 1 hour volatility measurement for the ETH/USDT trading pair.",
            inference_mode=og.InferenceMode.VANILLA,
        )

        return tool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return input_getter()


class TestParrotMultiplyToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[OpenGradientTool]:
        return OpenGradientTool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"a": 2, "b": 3}
