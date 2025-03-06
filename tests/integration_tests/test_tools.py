from enum import Enum

import opengradient as og
import pytest
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ToolsIntegrationTests
from opengradient import InferenceResult, ModelOutput
from pydantic import BaseModel, Field

from langchain_opengradient.toolkits import OpenGradientToolkit
from typing import Any, Dict


@pytest.mark.usefixtures("mock_env")
class TestOpenGradientRunModelNoSchemaToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        # model_input_provider has no inputs, so tool_input_schema not necessary
        def model_input_provider() -> dict:
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

        def output_formatter(inference_result: InferenceResult) -> str:
            return format(float(inference_result.model_output["Y"].item()), ".3%")

        toolkit = OpenGradientToolkit()
        tool = toolkit.create_run_model_tool(
            model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",
            tool_name="one_hour_eth_usdt_volatility",
            model_input_provider=model_input_provider,
            model_output_formatter=output_formatter,
            tool_description="Generate the live 1 hour volatility measurement for the "
            "ETH/USDT trading pair.",
            inference_mode=og.InferenceMode.VANILLA,
        )

        return tool

    @property
    def tool_invoke_params_example(self) -> dict:
        """This tool call has no arguments."""
        return {}


class Token(str, Enum):
    """Example enum used to help create tool schema."""

    ETH = "ethereum"
    BTC = "bitcoin"


@pytest.mark.usefixtures("mock_env")
class TestOpenGradientRunModelWithSchemaToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        class VolatilityInputSchema(BaseModel):
            token: Token = Field(description="Token name specified by user.")

        # These functions would normally get live data from an exchange.
        def get_eth_data() -> dict:
            return {"price_series": [2010.1, 2012.3, 2020.1, 2019.2]}

        def get_btc_data() -> dict:
            return {"price_series": [100001.1, 100013.2, 100149.2, 99998.1]}

        def model_input_provider(**llm_input: Any) -> dict:
            token = llm_input.get("token")
            if token == Token.BTC:
                return get_btc_data()
            elif token == Token.ETH:
                return get_eth_data()
            else:
                raise ValueError("Unexpected option found")

        def output_formatter(inference_result: InferenceResult) -> str:
            return format(float(inference_result.model_output["std"].item()), ".3%")

        toolkit = OpenGradientToolkit()
        tool = toolkit.create_run_model_tool(
            model_cid="QmZdSfHWGJyzBiB2K98egzu3MypPcv4R1ASypUxwZ1MFUG",
            tool_name="Return_volatility_tool",
            model_input_provider=model_input_provider,
            model_output_formatter=output_formatter,
            tool_input_schema=VolatilityInputSchema,
            tool_description="This tool takes a token and measures the return "
            "volatility (standard deviation of returns).",
            inference_mode=og.InferenceMode.VANILLA,
        )

        return tool

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns the example input argument of a token enum representing BTC.
        """
        return {"token": Token.BTC}


@pytest.mark.usefixtures("mock_env")
class TestOpenGradientReadWorkflowIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> BaseTool:
        def format_output(model_output: ModelOutput) -> str:
            value = format(
                float(model_output.numbers["regression_output"].item()), ".10%"
            )
            return f"Project change in price for ETH in the next hour is {value}"

        toolkit = OpenGradientToolkit()
        tool = toolkit.create_read_workflow_tool(
            workflow_contract_address="0x58826c6dc9A608238d9d57a65bDd50EcaE27FE99",
            tool_name="ETH_Price_Forecast",
            tool_description="Reads latest forecast for ETH price",
            output_formatter=format_output,
        )

        return tool

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Read workflow tool calls do not take in any tool inputs. The smart contract
        addresses that are being read from are already hard-coded in the tool
        definition.
        """
        return {}
