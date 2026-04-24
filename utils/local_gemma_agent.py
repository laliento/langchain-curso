from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGenerationChunk, ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage, AIMessageChunk
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Optional, List, Any, Dict, Type, Union
from pydantic import BaseModel
import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain_core.utils.function_calling import convert_to_openai_tool
import os
import sys
# Buscamos la raíz del proyecto (un nivel arriba de donde estamos)
from pathlib import Path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)
from agentes_pruebas.config_agent import AGENT_DESCRIPTIONS as desc_map
# Suprimir warnings de Transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configurar logging para suprimir warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

_SHARED_MODEL = None
_SHARED_PROCESSOR = None
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
        global _SHARED_MODEL, _SHARED_PROCESSOR
        
        if _SHARED_MODEL is None:
            from transformers import AutoProcessor, AutoModelForCausalLM
            print(f"--- CARGANDO PESOS POR ÚNICA VEZ ---")
            _SHARED_PROCESSOR = AutoProcessor.from_pretrained(self.model_path)
            _SHARED_MODEL = AutoModelForCausalLM.from_pretrained(self.model_path, 
                                                               dtype=self.dtype, 
                                                               device_map=self.device)
            _SHARED_MODEL.eval()
        
        self.model = _SHARED_MODEL
        self.processor = _SHARED_PROCESSOR

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 0. Identificar rol
        is_supervisor = any("transfer_to" in t.get("function", {}).get("name", "") for t in (self.tools or []))
        current_role = "SUPERVISOR" if is_supervisor else "ESPECIALISTA"
        print(f"\n>>> EJECUTANDO LLM COMO: {current_role} <<<")

        """Generate responses for a list of messages."""
        import json
        import re
        
        # 1. Preparar mensajes
        messages_dict = []
        
        # Detectar si estamos en modo tool calling (hay tool_calls en el historial)
        has_tool_calls = any(
            hasattr(m, 'tool_calls') and m.tool_calls 
            for m in messages
        )
        
        # Detectar si es respuesta de herramienta
        is_tool_response = any(getattr(m, 'tool_call_id', None) for m in messages)
        tools_formatted = ""
        if is_supervisor:
            tools_desc = []
            for t in self.tools:
                func_info = t.get("function", {})
                real_name = func_info.get("name") # ej. transfer_to_matematico
                clean_name = real_name.replace("transfer_to_", "")
                desc = desc_map.get(clean_name, "Agente especializado")
                tools_desc.append(f"- AGENT: {real_name}\n  Especialidad: {desc}")
                tools_formatted = "\n".join(tools_desc)
            tools_desc = []
            #for t in self.tools:
            #    func_info = t.get("function", {})
            #    real_name = func_info.get("name") # ej. transfer_to_matematico
            #    
            #    # Intentamos recuperar una descripción más rica
            #    # Si el supervisor no la tiene, podemos 'limpiar' el nombre para el prompt
            #    clean_name = real_name.replace("transfer_to_", "")
            #    desc = custom_desc = "Busca información general, cultural o de hechos históricos pasados sobre un tema específico." if "investigador" in clean_name else "Resuelve sólo operaciones matemáticas complejas." # desc_map.get(clean_name, "Agente especializado")
            #    
            #    tools_desc.append(f"- AGENT: {real_name}\n  Especialidad: {desc}")
            #    
            #    tools_formatted = "\n".join(tools_desc)
        for msg in messages:
            if isinstance(msg, HumanMessage):
                if is_supervisor:
                    messages_dict.append({"role": "user", "content": msg.content})
                elif tools_formatted and not is_tool_response: # si es humano, pero que no sea el original
                    messages_dict.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                # Si el AIMessage tiene tool_calls, no incluimos el contenido en el prompt
                # porque ya fue procesado como tool call
                if is_supervisor:
                    if not msg.tool_calls:
                        agent_name = msg.name if msg.name else "unknown"
                        full_name = f"transfer_to_{agent_name}"
                        content = msg.content
                        resultado = f"Respuesta obtenida de AGENT {full_name}: {content}"
                        messages_dict.append({"role": "assistant", "content": resultado})
                if not is_supervisor:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        first_tool_call = msg.tool_calls[0]
                        args = first_tool_call.get('args', {})
                        # Verificamos que 'input' exista en los argumentos y que no sea un handoff
                        if 'input' in args and first_tool_call.get('name') != 'transfer_back_to_supervisor':
                            specific_input = args.get('input')
                            print(f"DEBUG - Input detectado: {specific_input}")
                            messages_dict.append({"role": "user", "content": specific_input})
                        #else:
                        #    print("DEBUG - Mensaje descartado (es un handoff o no tiene input)")
                        #    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        #        # Agregamos un mensaje indicando que se usaron herramientas
                        #        tool_names = ", ".join([tc.get("name", "unknown") for tc in msg.tool_calls])
                        #        messages_dict.append({
                        #            "role": "assistant", 
                        #            "content": f"[Usé herramientas: {tool_names}]"
                        #        })
                        #    else:
                        #        messages_dict.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                if tools_formatted:
                    content = msg.content.replace("__TOOLS__", tools_formatted)
                    messages_dict.append({"role": "system", "content": content})
                else:
                    messages_dict.append({"role": "system", "content": msg.content})
            #elif isinstance(msg, ToolMessage):
            #    messages_dict.append({
            #        "role": "tool", 
            #        "tool_call_id": msg.tool_call_id,
            #        "content": msg.content
            #    })
        # --- SENSOR DE CONTEXTO ---
        print("\n" + "="*50)
        print("PROMPT ENVIADO AL MODELO (MESSAGES_DICT):")
        for m in messages_dict:
            print(f"[{m['role'].upper()}]: {m['content']}")
        print("="*50 + "\n")
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
        }
        # Solo enviamos flags de muestreo si la temperatura es mayor a 0
        if self.temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
            generate_kwargs["top_k"] = self.top_k
        else:
            # Si la temperatura es 0, forzamos Greedy Decoding
            generate_kwargs["do_sample"] = False

        # Fusionar con kwargs adicionales (para compatibilidad con LangGraph)
        generate_kwargs.update({k: v for k, v in kwargs.items() if k not in ['top_p', 'top_k']})

        outputs = self.model.generate(**inputs, **generate_kwargs)
        
        # 3. DECODIFICAR
        response = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()
        
        # --- DEBUG: Ver qué responde el modelo exactamente ---
        print(f"\n[DEBUG - Respuesta RAW del modelo]:\n{response}\n")

        # 4. Crear el mensaje de AI
        ai_message = AIMessage(content=response)
        
        if self.tools and is_supervisor:
            tool_calls = []
            pattern = r"DELEGAR_A_AGENTE:\s*(transfer_to_[a-z_]+)\s*\|\s*INPUT:\s*(.*)"
            matches = re.findall(pattern, response, re.IGNORECASE)
            for i, (t_name, t_input) in enumerate(matches):
                print(f"[DEBUG - Comando detectado]: Agente: {t_name} | Input: {t_input}")
                
                tool_calls.append({
                    "name": t_name.strip(),
                    "args": {"input": t_input.strip()},
                    "id": f"call_{os.urandom(4).hex()}",
                    "type": "tool_call"
                })
            
            # Intentar parsear como array JSON primero (múltiples tool calls)
            #clean_response = response.strip()
            #if clean_response.startswith("```json"):
            #    clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            #elif clean_response.startswith("```"):
            #    clean_response = clean_response.replace("```", "").strip()
            #tool_calls = []
            #try:
            #    # Intentar parsear como array
            #    data = json.loads(clean_response)
            #    if isinstance(data, list):
            #        # Es un array de tool calls
            #        matches = data
            #    elif isinstance(data, dict) and "name" in data:
            #        # Es un solo tool call
            #        matches = [data]
            #    else:
            #        matches = []
            #except Exception as e:
            #    # Si falla, intentar con regex
            #    # Buscar bloques JSON que tengan "name" y "arguments"
            #    print(f"[DEBUG] json.loads falló: {e}. Intentando Regex: {clean_response}")
            #    matches = re.findall(r'\{[\s\S]*?"name"[\s\S]*?"arguments"[\s\S]*?\}', response)
            #
            #print(f"[DEBUG - Coincidencias Regex]: {matches}")
            #
            #for i, match in enumerate(matches):
            #    try:
            #        if isinstance(match, dict):
            #            clean_match = match
            #        else:
            #            clean_match = match.strip()
            #            clean_match = json.loads(clean_match)
            #        
            #        t_name = clean_match.get("name")
            #        t_args = clean_match.get("arguments", {})
            #        
            #        # Asegurar que t_args sea un dict
            #        if isinstance(t_args, str):
            #            t_args = {"input": t_args}
            #        
            #        if t_name:
            #            print(f"[DEBUG - Tool Call {i} corregida]: {t_name} con args {t_args}")
            #            tool_calls.append({
            #                "name": t_name,
            #                "args": t_args,
            #                "id": f"call_{os.urandom(4).hex()}",
            #                "type": "tool_call"
            #            })
            #        else:
            #            print(f"[DEBUG - Tool Call {i} sin nombre válido]: {clean_match}")
            #    except Exception as e:
            #        print(f"[DEBUG - Error en Match {i}]: {e} | Contenido fallido: {match}")
            #        continue
            if tool_calls:
                ai_message.tool_calls = tool_calls
                ai_message.content = ""  # Vaciamos para que LangGraph ejecute las herramientas
            #else:
            #    # Si no hubo comandos de LLAMAR, es que el modelo escribió la respuesta final
            #    ai_message.content = response
            #if tool_calls:
            #    ai_message.tool_calls = tool_calls
            #    ai_message.content = ""  # Vaciar para que el supervisor actúe
            #    print(f"[DEBUG - Total Tool Calls inyectadas]: {len(tool_calls)}")

        return ChatResult(generations=[ChatGeneration(message=ai_message)])
    
    @property
    def _llm_type(self) -> str:
        return "local-gemma-4"
    
    def copy(self, **kwargs: Any) -> "LocalGemma4":
        """Create a copy of the model with updated parameters."""
        # Obtener los parámetros actuales
        params = {
            "model_path": self.model_path,
            "device": self.device,
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_thinking": self.enable_thinking,
            "tools": self.tools,
        }
        params.update(kwargs)
        return LocalGemma4(**params)
    
    def __getstate__(self):
        """Handle pickling - exclude model and processor which can't be pickled."""
        state = self.__dict__.copy()
        # No picklear el modelo y processor (se recargarán)
        state['model'] = None
        state['processor'] = None
        return state
    
    def __setstate__(self, state):
        """Handle unpickling - reload model and processor."""
        self.__dict__.update(state)
        # Recargar el modelo
        self._load_model()
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Stream responses for messages."""
        # For simplicity, generate full response and yield it as a single chunk
        result = self._generate(messages, stop=stop, **kwargs)
        for chunk in result.generations:
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
