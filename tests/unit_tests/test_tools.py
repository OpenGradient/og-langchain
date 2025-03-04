"""Unit testing for the tools generated by the OpenGradient toolkit."""

from typing import List
from unittest.mock import MagicMock, patch

import opengradient as og
from langchain_core.tools import BaseTool
from langchain_tests.unit_tests import ToolsUnitTests
from pydantic import BaseModel, Field

from langchain_opengradient.toolkits import OpenGradientToolkit


class MockInputSchema(BaseModel):
    values: List[float] = Field(description="List of values to process")


def mock_input_getter() -> None:
    return {"example": "getter"}


def mock_output_formatter(response) -> str:
    return f"Processed result: {response}"


class TestOpenGradientRunModelToolUnit(ToolsUnitTests):
    """Unit tests for a tool created by OpenGradientToolkit.create_run_model_tool"""

    @property
    def tool_constructor(self) -> BaseTool:
        """
        Return an instance of a run_model_tool created by the OpenGradient toolkit
        for unit testing.
        """
        with patch("opengradient.init") as mock_init:
            mock_client = MagicMock()
            mock_init.return_value = mock_client

            toolkit = OpenGradientToolkit(private_key="test_key")

            tool = toolkit.create_run_model_tool(
                model_cid="QmTest123456789",
                tool_name="test_model_tool",
                input_getter=mock_input_getter,
                output_formatter=mock_output_formatter,
                input_schema=MockInputSchema,
                tool_description="Test model tool for unit testing",
                inference_mode=og.InferenceMode.VANILLA,
            )

            return tool

    @property
    def tool_constructor_params(self) -> dict:
        """Return the parameters needed to construct the tool"""
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Return example parameters for invoking the tool"""
        return {"values": [5.0, 6.0, 7.0, 8.0]}


class TestOpenGradientReadWorkflowToolUnit(ToolsUnitTests):
    """Unit tests for a tool created by OpenGradientToolkit.create_read_workflow_tool"""

    @property
    def tool_constructor(self) -> BaseTool:
        """
        Return an instance of a read_workflow_tool created by the OpenGradient toolkit
        for unit testing.
        """
        with patch("opengradient.init") as mock_init:
            mock_client = MagicMock()
            mock_init.return_value = mock_client

            toolkit = OpenGradientToolkit(private_key="test_key")

            tool = toolkit.create_read_workflow_tool(
                workflow_contract_address="0x123456789",
                tool_name="test_workflow_tool",
                tool_description="Test model tool for unit testing",
                output_formatter=mock_output_formatter,
            )

            return tool

    @property
    def tool_constructor_params(self) -> dict:
        """Return the parameters needed to construct the tool"""
        return {}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Return example parameters for invoking the tool.

        read_workflow type tools don't require any parameters.
        """
        return {}
