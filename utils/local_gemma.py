from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any, Dict, Type, Union
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain_core.utils.function_calling import convert_to_openai_tool
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
        import json
        import re
        
        # 1. Preparar mensajes y inyectar herramientas en el System Prompt
        messages_dict = []
        
        # Si hay herramientas vinculadas, las exponemos en el prompt
        if self.tools:
            tools_json = json.dumps(self.tools, indent=2)
            system_instruction = (
                "Eres un asistente que puede usar herramientas. "
                "Si decides usar una, DEBES responder EXCLUSIVAMENTE con un JSON: "
                '{"name": "nombre_de_la_herramienta", "arguments": {"arg": "valor"}}. '
                f"Herramientas disponibles:\n{tools_json}"
            )
            # Insertamos la instrucción al inicio
            messages_dict.append({"role": "system", "content": system_instruction})

        for msg in messages:
            if isinstance(msg, HumanMessage):
                messages_dict.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_dict.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages_dict.append({"role": "system", "content": msg.content})

        # 2. Aplicar plantilla y generar
        text = self.processor.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p

        outputs = self.model.generate(**inputs, **generate_kwargs)
        
        # 3. DECODIFICAR (Aquí es donde se define 'response')
        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()
        
        # 4. Crear el mensaje de AI y buscar Tool Calls
        ai_message = AIMessage(content=response)
        
        if self.tools:
            # Buscamos un patrón JSON en la respuesta
            json_match = re.search(r'\{[\s\S]*"name"[\s\S]*"arguments"[\s\S]*\}', response)
            if json_match:
                try:
                    # Limpiamos posibles ruidos antes/después del JSON
                    clean_json = json_match.group()
                    tool_data = json.loads(clean_json)
                    
                    # Formato oficial de LangChain para tool_calls
                    ai_message.tool_calls = [{
                        "name": tool_data["name"],
                        "args": tool_data["arguments"],
                        "id": f"call_{os.urandom(4).hex()}",
                        "type": "tool_call"
                    }]
                except Exception as e:
                    print(f"Error parseando tool_call: {e}")

        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
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
        tools: List[Any],
        **kwargs: Any,
    ):
        """Bind tools to the model correctly for LangChain."""
        # convert_to_openai_tool maneja automáticamente @tool, Pydantic models y dicts
        tools_dict = [convert_to_openai_tool(t) for t in tools]
        
        new_instance = self.copy()
        new_instance.tools = tools_dict
        return new_instance
    ## Opcion que funciona con pydantic
    def with_structured_output(self, schema: Type[BaseModel], **kwargs: Any):
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        
        # 1. El Parser oficial de LangChain que usa Pydantic
        parser = PydanticOutputParser(pydantic_object=schema)
        
        # 2. Creamos un template que inyecta las instrucciones de formato automáticamente
        # El parser tiene un método para obtener las instrucciones que el LLM entiende
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Responde siempre siguiendo este formato: {format_instructions}"),
            ("human", "{input}")
        ]).partial(format_instructions=parser.get_format_instructions())

        # 3. La cadena es "Natural": Prompt -> LLM -> Parser
        # El parser se encarga de llamar a Pydantic y validar
        return prompt | self | parser
    ## Opcion que funciona con pydantic
    #def with_structured_output(
    #    self, schema: Type[BaseModel], **kwargs: Any
    #):
    #    """Return a chain that structured output using tool calling."""
    #    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    #    from langchain_core.messages import SystemMessage
    #    import json
#
    #    # Extraemos el esquema JSON de Pydantic para el prompt
    #    schema_json = json.dumps(schema.model_json_schema(), indent=2)
#
    #    def add_schema_to_prompt(input_data):
    #        # Si el input es una cadena, la convertimos a mensaje
    #        content = input_data if isinstance(input_data, str) else str(input_data)
    #        
    #        instruction = (
    #            f"Analiza la solicitud del usuario y responde EXCLUSIVAMENTE con un objeto JSON "
    #            f"que cumpla con este esquema:\n{schema_json}\n"
    #            "No incluyas texto adicional antes ni después del JSON."
    #        )
    #        
    #        # Devolvemos una lista de mensajes para el LLM
    #        return [
    #            SystemMessage(content=instruction),
    #            HumanMessage(content=content)
    #        ]
#
    #    def extract_and_parse(message) -> BaseModel:
    #        import re
    #        import json
    #        
    #        content = message.content if hasattr(message, 'content') else str(message)
    #        
    #        # Búsqueda de JSON más agresiva
    #        json_match = re.search(r'\{[\s\S]*\}', content)
    #        
    #        if json_match:
    #            try:
    #                data = json.loads(json_match.group())
    #                return schema(**data)
    #            except Exception as e:
    #                print(f"Error de validación Pydantic: {e}")
    #                print(f"Content recibido: {content}")
    #        
    #        # CAMBIO CRUCIAL: En lugar de fallar con schema(**{}), 
    #        # devolvemos el error o un objeto con valores de error para no romper la cadena.
    #        # O mejor aún, puedes lanzar una excepción clara.
    #        raise ValueError(f"El modelo no generó un JSON válido para el esquema {schema.__name__}. Respuesta: {content}")
    #    
    #    # La cadena ahora: 
    #    # 1. Transforma el texto en mensajes con instrucciones de esquema.
    #    # 2. El modelo genera la respuesta.
    #    # 3. El parser extrae el objeto Pydantic.
    #    return RunnableLambda(add_schema_to_prompt) | self | RunnableLambda(extract_and_parse)
