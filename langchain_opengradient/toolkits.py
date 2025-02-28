"""OpenGradient toolkits."""

from typing import List, Callable, Type, Optional
from pydantic import BaseModel
import os

from langchain_core.tools import BaseTool, BaseToolkit
from opengradient.alphasense import create_run_model_tool, create_read_workflow_tool, ToolType
import opengradient as og

class OpenGradientToolkit(BaseToolkit):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/c123cb2b304f52ab65db4714eeec46af69a861ec/libs/community/langchain_community/agent_toolkits/sql/toolkit.py#L19
    """OpenGradient toolkit.

    # TODO: Replace with relevant packages, env vars, etc.
    Setup:
        Install ``langchain-opengradient`` and set environment variable ``OPENGRADIENT_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-opengradient
            export OPENGRADIENT_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args:
        arg 1: type
            description
        arg 2: type
            description

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain-opengradient import OpenGradientToolkit

            toolkit = OpenGradientToolkit(
                # ...
            )

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        .. code-block:: none

            # TODO: Example output.

    Use within an agent:
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent

            agent_executor = create_react_agent(llm, tools)

            example_query = "..."

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             # TODO: Example output.

    """  # noqa: E501

    def __init__(self):
        # Initialize OpenGradient client
        private_key = private_key or os.getenv("OPENGRADIENT_PRIVATE_KEY")
        if not private_key:
            raise ValueError("OPENGRADIENT_PRIVATE_KEY environment variable is not set")

        og.init(private_key=private_key, email=None, password=None)

        # List of tool (Expected to be of type BaseTool)
        self.tools = []

    # TODO: This method must be implemented to list tools.
    def get_tools(self) -> List[BaseTool]:
        """Get list of tools available in OpenGradient toolkit."""
        return self.tools
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add tool to the list of tools for the OpenGradient Agentkit."""
        # Maybe add a check here
        self.tools.append(tool)

    def create_run_model_tool(self,
                              model_cid: str,
                              tool_name: str,
                              input_getter: Callable,
                              output_formatter: Callable[..., str] = lambda x: x,
                              input_schema: Optional[Type[BaseModel]] = None,
                              tool_description: str = "Executes the given ML model",
                              inference_mode: og.InferenceMode = og.InferenceMode.VANILLA) -> BaseTool:
        """
        Wrapper for create_run_model_tool from OpenGradient AlphaSense library.
        
        This function creates a langchain compatible tool to run inferences on the OpenGradient network. 

        Example usage:
            from og_langchain.toolkits import OpenGradientToolkit

            TODO(Kyle): Fill me in

        """
        tool = create_run_model_tool(tool_type=ToolType.LANGCHAIN, 
                                     model_cid=model_cid, 
                                     tool_name=tool_name, 
                                     input_getter=input_getter, 
                                     output_formatter=output_formatter, 
                                     input_schema=input_schema,
                                     tool_description=tool_description,
                                     inference_mode=inference_mode)
        
        return tool

    def create_read_workflow_tool(self,
                                  workflow_contract_address: str,
                                  tool_name: str,
                                  tool_description: str,
                                  output_formatter: Callable[..., str] = lambda x: x) -> BaseTool:
        """
        Wrapper for create_read_workflow_tool from OpenGradient AlphaSense library.
        
        This function creates a langchain compatible tool to read workflows on the OpenGradient network. 

        Example usage:
            from og_langchain.toolkits import OpenGradientToolkit
            
            toolkit = OpenGradientToolkit()
            btc_workflow_tool = toolkit.create_read_workflow_tool(
                tool_type=ToolType.LANGCHAIN,
                workflow_contract_address="0x6e0641925b845A1ca8aA9a890C4DEF388E9197e0",
                tool_name="ETH_Price_Forecast",
                tool_description="Reads latest forecast for ETH price",
                output_formatter=lambda x: x,
            )

            toolkit.add_tool(btc_workflow_tool)

            for tool in toolkit.get_tools():
                print(tool)
        """
        tool = create_read_workflow_tool(tool_type=ToolType.LANGCHAIN,
                                         workflow_contract_address=workflow_contract_address,
                                         tool_name=tool_name,
                                         tool_description=tool_description,
                                         output_formatter=output_formatter)
        
        return tool
