from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any, Dict, Type, Union
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


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
    tools: Optional[List[Dict[str, Any]]] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model()
    
    def _load_model(self):
        """Load the model from local path."""
        from pathlib import Path
        
        # Convert to absolute path
        model_path = Path(self.model_path).resolve()
        
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            device_map=self.device,
            local_files_only=True
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
    
    def bind_tools(
        self,
        tools: List[Union[Dict[str, Any], Type[BaseModel], Callable[..., Any]]],
        **kwargs: Any,
    ):
        """Bind tools to the model."""
        # Convert tools to dict format
        tools_dict = []
        for tool in tools:
            if isinstance(tool, dict):
                tools_dict.append(tool)
            elif hasattr(tool, "model_json_schema"):
                # Pydantic model
                schema = tool.model_json_schema()
                tools_dict.append({
                    "type": "function",
                    "function": {
                        "name": schema.get("title", "tool"),
                        "description": schema.get("description", ""),
                        "parameters": schema.get("properties", {})
                    }
                })
            else:
                tools_dict.append(tool)
        
        new_instance = self.copy()
        new_instance.tools = tools_dict
        return new_instance
    
    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs: Any
    ):
        """Return a chain that structured output using tool calling."""
        from langchain_core.runnables import RunnableLambda
        
        # Create a parser that extracts JSON from text and converts to schema
        def extract_and_parse(message) -> BaseModel:
            """Extract JSON from message content and parse to schema."""
            import re
            import json
            
            # Get content from message
            content = message.content if hasattr(message, 'content') else str(message)
            
            # Try to find JSON in the response - more flexible pattern
            # Look for content between curly braces, including nested ones
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    return schema(**data)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Content: {content}")
            
            # If no JSON found, return empty schema with defaults
            return schema(**{})
        
        return RunnablePassthrough() | self | RunnableLambda(extract_and_parse)
