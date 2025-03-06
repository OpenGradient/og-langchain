"""Unit testing for the OpenGradient toolkit functions."""

from types import Any, Dict
from unittest.mock import patch

import opengradient as og
import pytest
from langchain_core.tools import BaseTool
from opengradient import InferenceResult, ModelOutput
from opengradient.alphasense import (
    ToolType,
)
from pydantic import BaseModel, Field

from langchain_opengradient.toolkits import OpenGradientToolkit


class MockTool(BaseTool):
    """Mocktool that inherits from Basetool for unit tests."""

    def __init__(self) -> None:
        return

    def _run(self) -> int:
        return 0


def test_toolkit_initialization_error() -> None:
    """Test that toolkit does not initialize without OpenGradient API key."""
    with pytest.raises(
        ValueError, match="OPENGRADIENT_PRIVATE_KEY environment variable is not set"
    ):
        OpenGradientToolkit()


@pytest.mark.usefixtures("mock_env")
def test_toolkit_initialization_success() -> None:
    """Test that toolkit initializes properly."""
    toolkit = OpenGradientToolkit()

    assert toolkit.get_tools() == []


@pytest.mark.usefixtures("mock_env")
def test_add_tool() -> None:
    """Test that tools can be added and are returned by the get_tools method."""
    toolkit = OpenGradientToolkit()

    tool = MockTool()
    toolkit.add_tool(tool)
    assert len(toolkit.get_tools()) == 1
    assert toolkit.get_tools() == [tool]


@pytest.mark.usefixtures("mock_env")
def test_create_run_model_tool_error() -> None:
    """Test error flow with function create_run_model_tool."""

    class ExampleInputSchema(BaseModel):
        example_int_field: int = Field(description="This is an example int field")
        example_str_field: str = Field(description="This is an example str field")

    model_cid = "Example_CID"
    tool_name = "Example run model tool"

    def model_input_provider(data: Any) -> Dict:
        return {"input": "example input getter function"}

    def model_output_formatter(output: InferenceResult) -> str:
        return str(output)

    tool_description = "This tool is an example tool."
    inference_mode = og.InferenceMode.VANILLA

    with patch(
        "langchain_opengradient.toolkits.create_run_model_tool"
    ) as mock_create_run_model_tool:
        # Set up the mock to raise an exception
        mock_create_run_model_tool.side_effect = ValueError("Invalid model CID")

        toolkit = OpenGradientToolkit()

        # Test that the error from create_run_model_tool is propagated
        with pytest.raises(ValueError, match="Invalid model CID"):
            toolkit.create_run_model_tool(
                model_cid=model_cid,
                tool_name=tool_name,
                model_input_provider=model_input_provider,
                model_output_formatter=model_output_formatter,
                tool_input_schema=ExampleInputSchema,
                tool_description=tool_description,
                inference_mode=inference_mode,
            )

        # Verify the mock was called with correct arguments
        mock_create_run_model_tool.assert_called_once_with(
            tool_type=ToolType.LANGCHAIN,
            model_cid=model_cid,
            tool_name=tool_name,
            model_input_provider=model_input_provider,
            model_output_formatter=model_output_formatter,
            tool_input_schema=ExampleInputSchema,
            tool_description=tool_description,
            inference_mode=inference_mode,
        )


@pytest.mark.usefixtures("mock_env")
def test_create_run_model_tool_success() -> None:
    """Test that create_run_model_tool returns a Langchain compatible tool."""

    class ExampleInputSchema(BaseModel):
        example_int_field: int = Field(description="This is an example int field")
        example_str_field: str = Field(description="This is an example str field")

    model_cid = "Example_CID"
    tool_name = "Example run model tool"

    def model_input_provider(data: Any) -> Dict:
        return {"input": "example input getter function"}

    def model_output_formatter(output: InferenceResult) -> str:
        return str(output)

    tool_description = "This tool is an example tool."
    inference_mode = og.InferenceMode.TEE

    with patch(
        "langchain_opengradient.toolkits.create_run_model_tool"
    ) as mock_create_run_model_tool:
        mock_create_run_model_tool.return_value = MockTool

        toolkit = OpenGradientToolkit()
        tool = toolkit.create_run_model_tool(
            model_cid=model_cid,
            tool_name=tool_name,
            model_input_provider=model_input_provider,
            model_output_formatter=model_output_formatter,
            tool_input_schema=ExampleInputSchema,
            tool_description=tool_description,
            inference_mode=inference_mode,
        )

        mock_create_run_model_tool.assert_called_once_with(
            tool_type=ToolType.LANGCHAIN,
            model_cid=model_cid,
            tool_name=tool_name,
            model_input_provider=model_input_provider,
            model_output_formatter=model_output_formatter,
            tool_input_schema=ExampleInputSchema,
            tool_description=tool_description,
            inference_mode=inference_mode,
        )
        assert tool == MockTool


@pytest.mark.usefixtures("mock_env")
def test_create_read_workflow_tool_error() -> None:
    """Test error flow with function create_read_workflow_tool."""
    workflow_contract_address = "0x12345"
    tool_name = "Example read workflow tool"

    def output_formatter(output: ModelOutput) -> str:
        return str(output)

    tool_description = "This tool is an example tool."

    with patch(
        "langchain_opengradient.toolkits.create_read_workflow_tool"
    ) as mock_create_read_workflow_tool:
        mock_create_read_workflow_tool.side_effect = ValueError(
            "Invalid workflow contract address"
        )

        toolkit = OpenGradientToolkit()

        # Test that the error from create_read_workflow_tool is propagated
        with pytest.raises(ValueError, match="Invalid workflow contract address"):
            toolkit.create_read_workflow_tool(
                workflow_contract_address=workflow_contract_address,
                tool_name=tool_name,
                output_formatter=output_formatter,
                tool_description=tool_description,
            )

        # Verify the mock was called with correct arguments
        mock_create_read_workflow_tool.assert_called_once_with(
            tool_type=ToolType.LANGCHAIN,
            workflow_contract_address=workflow_contract_address,
            tool_name=tool_name,
            output_formatter=output_formatter,
            tool_description=tool_description,
        )


@pytest.mark.usefixtures("mock_env")
def test_create_read_workflow_tool_success() -> None:
    """Test that create_read_workflow_tool returns a Langchain compatible tool."""
    workflow_contract_address = "0x12345"
    tool_name = "Example read workflow tool"

    def output_formatter(output: ModelOutput) -> str:
        return str(output)

    tool_description = "This tool is an example tool."

    with patch(
        "langchain_opengradient.toolkits.create_read_workflow_tool"
    ) as mock_create_read_workflow_tool:
        mock_create_read_workflow_tool.return_value = MockTool

        toolkit = OpenGradientToolkit()
        tool = toolkit.create_read_workflow_tool(
            workflow_contract_address=workflow_contract_address,
            tool_name=tool_name,
            output_formatter=output_formatter,
            tool_description=tool_description,
        )

        mock_create_read_workflow_tool.assert_called_once_with(
            tool_type=ToolType.LANGCHAIN,
            workflow_contract_address=workflow_contract_address,
            tool_name=tool_name,
            output_formatter=output_formatter,
            tool_description=tool_description,
        )
        assert tool == MockTool
