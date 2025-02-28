from importlib import metadata

from langchain_opengradient.chat_models import ChatOpenGradient
from langchain_opengradient.document_loaders import OpenGradientLoader
from langchain_opengradient.embeddings import OpenGradientEmbeddings
from langchain_opengradient.retrievers import OpenGradientRetriever
from langchain_opengradient.toolkits import OpenGradientToolkit
from langchain_opengradient.tools import OpenGradientTool
from langchain_opengradient.vectorstores import OpenGradientVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatOpenGradient",
    "OpenGradientVectorStore",
    "OpenGradientEmbeddings",
    "OpenGradientLoader",
    "OpenGradientRetriever",
    "OpenGradientToolkit",
    "OpenGradientTool",
    "__version__",
]
