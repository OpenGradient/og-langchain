"""Test OpenGradient embeddings."""

from typing import Type

from langchain_opengradient.embeddings import OpenGradientEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[OpenGradientEmbeddings]:
        return OpenGradientEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
