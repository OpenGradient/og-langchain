# langchain-opengradient

This package contains the LangChain integration with OpenGradient

## Installation

```bash
pip install -U langchain-opengradient
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatOpenGradient` class exposes chat models from OpenGradient.

```python
from langchain_opengradient import ChatOpenGradient

llm = ChatOpenGradient()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OpenGradientEmbeddings` class exposes embeddings from OpenGradient.

```python
from langchain_opengradient import OpenGradientEmbeddings

embeddings = OpenGradientEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`OpenGradientLLM` class exposes LLMs from OpenGradient.

```python
from langchain_opengradient import OpenGradientLLM

llm = OpenGradientLLM()
llm.invoke("The meaning of life is")
```
