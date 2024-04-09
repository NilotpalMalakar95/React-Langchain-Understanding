from uuid import UUID
from typing import List, Dict, Any, Optional

from langchain_core.outputs import LLMResult
from langchain.callbacks.base import BaseCallbackHandler


class AgentCallbacksHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print("*" * 100)
        print("The prompts to the LLM were :")
        print(prompts[0])
        print("*" * 100)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print("*" * 100)
        print("The response made by the LLM was :")
        print(response.generations[0][0].text)
        print("*" * 100)
