
import pytest
from langchain_core.tools import BaseTool

from langchain_opengradient.toolkits import OpenGradientToolkit


def test_toolkit_initialization_error():
    """Test that toolkit does not initialize without OpenGradient API key."""
    with pytest.raises(
        ValueError, match="OPENGRADIENT_PRIVATE_KEY environment variable is not set"
    ):
        OpenGradientToolkit()


@pytest.mark.usefixtures("mock_env")
def test_toolkit_initialization_success():
    """Test that toolkit initializes properly."""
    toolkit = OpenGradientToolkit()

    assert toolkit.get_tools() == []


@pytest.mark.usefixtures("mock_env")
def test_add_tool():
    """Test that tools can be added and are returned by the get_tools method."""
    toolkit = OpenGradientToolkit()

    class MockTool(BaseTool):
        def __init__(self):
            return

        def _run(self):
            return 0

    tool = MockTool()

    toolkit.add_tool(tool)
    assert len(toolkit.get_tools()) == 1
    assert toolkit.get_tools() == [tool]


@pytest.mark.usefixtures("mock_env")
def test_create_run_model_tool_error():
    """Test error flow with function create_run_model_tool."""
    pass


@pytest.mark.usefixtures("mock_env")
def test_create_run_model_tool_success():
    """Test that create_run_model_tool returns a Langchain compatible tool."""
    pass


@pytest.mark.usefixtures("mock_env")
def test_create_read_workflow_tool_error():
    """Test error flow with function create_read_workflow_tool."""
    pass


@pytest.mark.usefixtures("mock_env")
def test_create_read_workflow_tool_success():
    """Test that create_read_workflow_tool returns a Langchain compatible tool."""
    pass
