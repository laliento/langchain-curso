from langchain_core.language_models import BaseLLM
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any, Dict, Type
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class LocalGemma4(BaseLLM):
    """Custom LangChain LLM wrapper for local Gemma 4 model."""
    
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
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for a list of prompts."""
        generations = []
        
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            
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
            
            generations.append([GenerationChunk(text=response)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "local-gemma-4"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        """Call the LLM with a single prompt."""
        result = self._generate([prompt], stop=stop, **kwargs)
        return result.generations[0][0].text
    
    def with_structured_output(
        self, schema: Type[BaseModel], **kwargs: Any
    ):
        """Return a chain that structured output."""
        parser = PydanticOutputParser(pydantic_object=schema)
        return RunnablePassthrough() | self | parser
