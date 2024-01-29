from agentverse.registry import Registry

code_completion_registry = Registry(name="CodeCompletionRegistry")

from .base import BaseCodeCompletion, NoneCodeCompletion
from .basic import BasicCodeCompletion
from .RAG import RAGCodeCompletion
from .RAGwithDebug import RAGwithDebugCodeCompletion