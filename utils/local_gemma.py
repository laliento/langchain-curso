from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any, Dict, Type
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class LocalGemma4(BaseChatModel):
    """Custom LangChain ChatModel wrapper for local Gemma 4 model."""
    
    model_path: str
    model: Any = None
    processor: Any = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    enable_thinking: bool = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the model from local path."""
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            device_map=self.device
        )
        self.model.eval()
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate responses for a list of messages."""
        generations = []
        
        # Convert messages to dict format
        messages_dict = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                messages_dict.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_dict.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages_dict.append({"role": "system", "content": msg.content})
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # Handle temperature=0 for greedy decoding
        if self.temperature == 0:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        
        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )
        
        ai_message = AIMessage(content=response)
        generations.append(ChatGeneration(message=ai_message))
        
        return ChatResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "local-gemma-4"
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Stream responses for messages."""
        # For simplicity, generate full response and yield it as a single chunk
        result = self._generate([messages], stop=stop, **kwargs)
        for chunk in result.generations[0]:
            yield chunk
    
    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs: Any
    ):
        """Return a chain that structured output."""
        parser = PydanticOutputParser(pydantic_object=schema)
        return RunnablePassthrough() | self | parser
