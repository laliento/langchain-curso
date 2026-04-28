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
            print(f"--- CARGANDO MODELO POR ÚNICA VEZ ---")
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
        import json
        import re
        # --- INICIO DEL DEBUGGER ---

        # 0. Identificar rol
        is_supervisor = any("transfer_to" in t.get("function", {}).get("name", "") for t in (self.tools or []))
        ya_tiene_respuesta_tool = False
        esperando_ejecucion_tool = False
        if not is_supervisor:
            last_msg = messages[-1]
            if isinstance(last_msg, ToolMessage):
                # Si el último mensaje es ToolMessage, revisamos si es de una función real o un transfer
                content_str = str(last_msg.content)
                
                if "Successfully transferred" in content_str:
                    # Acabamos de llegar del supervisor, el LLM debe actuar por primera vez
                    # --- EXTRACCIÓN DE METADATOS DE LA TOOL ---
                    if self.tools and len(self.tools) > 0:
                        # Tomamos la primera herramienta (asumiendo que el especialista tiene una principal)
                        # O puedes buscarla por nombre si tuviera varias
                        tool_info = self.tools[0].get("function", {})
                        
                        tool_name = tool_info.get("name")
                        tool_description = tool_info.get("description")

                        # 1. Entramos a la definición de la función
                        function_def = self.tools[0].get("function", {})
                        # 2. Entramos a 'parameters' y luego a 'properties'
                        properties = function_def.get("parameters", {}).get("properties", {})
                        
                        if properties:
                            # Obtenemos el nombre del primer parámetro (ej. 'expresion')
                            param_name = list(properties.keys())[0]
                            
                            # Obtenemos su tipo (ej. 'string')
                            param_type = properties[param_name].get("type")
                            
                            print(f"🛠 PARÁMETRO DETECTADO: {param_name} ({param_type})")
                    # ------------------------------------------
                    esperando_ejecucion_tool = True
                    current_role = "ESPECIALISTA_INIT"
                    print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EJECUTANDO LLM COMO: {current_role} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    print("📌 ESTADO: Recién transferido. El LLM debe decidir qué herramienta usar.")
                    print(f"📦 INFO TOOL CARGADA: {tool_name}")
                    print(f"📖 DESCRIPCIÓN: {tool_description[:50]}...") # Print corto para no saturar
                else:
                    # Es una respuesta de una herramienta funcional (ej. "Resultado: 12")
                    ya_tiene_respuesta_tool = True
                    # CAPTURA AQUÍ EL CONTENIDO
                    resultado_de_la_tool = last_msg.content  # Aquí tendrás "Resultado: 12"
                    nombre_de_la_tool = last_msg.name        # Aquí tendrás "calcular_method"
                    current_role = "ESPECIALISTA_TOOL_RESPONSE"
                    print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EJECUTANDO LLM COMO: {current_role} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                    print(f"📌 ESTADO: Tooll '{nombre_de_la_tool}' devolvió: {resultado_de_la_tool} . El LLM debe procesar el resultado.")
            
            elif isinstance(last_msg, HumanMessage):
                # Caso en que el especialista es llamado directamente sin pasar por transfer (raro en tu flujo pero posible)
                print("📌 ESTADO: Petición directa del usuario. El LLM debe decidir acción.")
        else:
            current_role = "SUPERVISOR"
            print(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EJECUTANDO LLM COMO: {current_role} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        """Generate responses for a list of messages."""
        # --- SCANNER DE TOOLS ---
        print("\n" + "🔍" * 20)
        print("REVISANDO TOOLS VINCULADAS AL MODELO:")
        is_tool_calcular = False
        is_tool_buscar_web = False
        if self.tools:
            print(f"Total de herramientas: {len(self.tools)}")
            for i, tool in enumerate(self.tools):
                # Imprimimos el nombre y los parámetros para verificar
                name = tool.get("function", {}).get("name", "Sin nombre")
                if name == "calcular_method":
                    is_tool_calcular = True
                elif name == "investigador_method":
                    is_tool_buscar_web = True
                print(f"Tool [{i}]: {name}")
                # Si quieres ver todo el esquema JSON de la tool:
                #print(json.dumps(tool, indent=2, ensure_ascii=False))
        else:
            print("AVISO: No hay herramientas (self.tools está vacío o es None)")
        print("🔍" * 20 + "\n")

        # --- DEBUGGER DE MENSAJES (Insertar aquí) ---
        print("\n" + "="*60)
        print(f"DEBUG: ENTRADA A LOCAL_GEMMA (Mensajes recibidos: {len(messages)})")
        print("="*60)
        
        for i, m in enumerate(messages):
            # Identificar el tipo de objeto
            tipo = type(m).__name__
            
            # Colores básicos para consola: System(Azul), Human(Verde), AI(Amarillo), Tool(Cian)
            color = "\033[94m" if tipo == "SystemMessage" else \
                    "\033[92m" if tipo == "HumanMessage" else \
                    "\033[93m" if tipo == "AIMessage" else \
                    "\033[96m" if tipo == "ToolMessage" else "\033[0m"
            reset = "\033[0m"
            
            print(f"{color}Mensaje [{i}] - Tipo: {tipo}{reset}")
            
            # Si es un AIMessage, ver si trae tool_calls (lo que genera el bucle)
            if hasattr(m, "tool_calls") and m.tool_calls:
                print(f"   └─ 🛠 Tool Calls: {m.tool_calls}")
            
            # Si es un ToolMessage, ver el ID para confirmar que matchea con el AI
            if tipo == "ToolMessage":
                print(f"   └─ 🆔 ID de la Tool: {m.tool_call_id}")
            
            # Imprimir una parte del contenido para no saturar
            snippet = (str(m.content)[:100] + '...') if len(str(m.content)) > 100 else m.content
            print(f"   └─ 📝 Contenido: {snippet}")
            
        print("="*60 + "\n")
        # --- FIN DEL DEBUGGER ---
        ################################### Preparar variables clave
        messages_dict = []
        tool_calls = []
        tool_name = ""
        tool_param_name = ""
        too_param_type = ""
        supervisor_input_to_agent = ""
        resultado_de_la_tool = ""
        final_agent_response = ""
        if is_supervisor:
            tools_desc = []
            tools_formatted = ""
            for t in self.tools:
                func_info = t.get("function", {})
                real_name = func_info.get("name") # ej. transfer_to_matematico
                clean_name = real_name.replace("transfer_to_", "")
                desc = desc_map.get(clean_name, "Agente especializado")
                tools_desc.append(f"- AGENT: {real_name}\n  Especialidad: {desc}")
                tools_formatted = "\n".join(tools_desc)
        elif esperando_ejecucion_tool:
            # --- EXTRACCIÓN DE METADATOS DE LA TOOL ---
            if self.tools and len(self.tools) > 0:
                # Tomamos la primera herramienta (asumiendo que el especialista tiene una principal)
                # O puedes buscarla por nombre si tuviera varias
                tool_info = self.tools[0].get("function", {})
                tool_name = tool_info.get("name")

                # 1. Entramos a la definición de la función
                function_def = self.tools[0].get("function", {})
                # 2. Entramos a 'parameters' y luego a 'properties'
                properties = function_def.get("parameters", {}).get("properties", {})
                
                if properties:
                    # Obtenemos el nombre del primer parámetro (ej. 'expresion')
                    tool_param_name = list(properties.keys())[0]
                    # Obtenemos su tipo (ej. 'string')
                    too_param_type = properties[param_name].get("type")
                    print(f"📦 INFO TOOL CARGADA: {tool_name} con PARÁMETRO DETECTADO: {param_name} ({param_type})")
            # Extraccion de los mensajes la instrucción específica para este agente que fue mandado por el supervisor
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    first_tool_call = msg.tool_calls[0]
                    args = first_tool_call.get('args', {})
                    # Verificamos que 'input' exista en los argumentos y que no sea un handoff
                    if 'input' in args and first_tool_call.get('name') != 'transfer_back_to_supervisor':
                        supervisor_input_to_agent = args.get('input')
                        print(f"DEBUG - Input detectado: {supervisor_input_to_agent}")
                        #messages_dict.append({"role": "user", "content": specific_input})
        elif ya_tiene_respuesta_tool:
            # Extraccion de los mensajes la instrucción específica para este agente que fue mandado por el supervisor
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    first_tool_call = msg.tool_calls[0]
                    args = first_tool_call.get('args', {})
                    # Verificamos que 'input' exista en los argumentos y que no sea un handoff
                    if 'input' in args and first_tool_call.get('name') != 'transfer_back_to_supervisor':
                        supervisor_input_to_agent = args.get('input')
                        print(f"DEBUG - Input detectado: {supervisor_input_to_agent}")
                        #messages_dict.append({"role": "user", "content": specific_input})
            # CAPTURA AQUÍ EL CONTENIDO
            last_msg = messages[-1]
            resultado_de_la_tool = last_msg.content  # Aquí tendrás "Resultado: 12"
            final_agent_response =f"{supervisor_input_to_agent} \n - Respuesta : {resultado_de_la_tool}"
        
        ################################### Preparar mensajes para LLM
        if is_supervisor:
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    if tools_formatted:
                        content = msg.content.replace("__TOOLS__", tools_formatted)
                        messages_dict.append({"role": "system", "content": content})
                    else:
                        messages_dict.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    messages_dict.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    # Respuesta de agente
                    if not msg.tool_calls:
                        agent_name = msg.name if msg.name else "unknown"
                        full_name = f"transfer_to_{agent_name}"
                        content = msg.content
                        resultado = f"Respuesta obtenida de AGENT {full_name}: \n - Pregunta: {content}"
                        messages_dict.append({"role": "assistant", "content": resultado})
        elif esperando_ejecucion_tool:
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    messages_dict.append({"role": "system", "content": msg.content})
            # mensaje para que de formato de la tool
            messages_dict.append({"role": "user", "content": supervisor_input_to_agent})# TODO
        elif ya_tiene_respuesta_tool:
            ai_message = AIMessage(content=final_agent_response)
            return ChatResult(generations=[ChatGeneration(message=ai_message)])
        
        ###################################Ejecución de LLM
        print("\n" + "="*50)
        print(f"PROMPT ENVIADO AL MODELO (MESSAGES_DICT): {len(messages_dict)}")
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


        ###################################Creación de tools
        if self.tools:
            if is_supervisor:
                ## añade tools que son los agentes
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
            elif esperando_ejecucion_tool:
                ## añade tools que son los métodos a ejecutar
                tool_calls.append({
                    "name":tool_name,
                    "args": {tool_param_name: response}, 
                    "id": f"call_{os.urandom(4).hex()}",
                    "type": "tool_call"
                })
            # add tools
            ai_message.tool_calls = tool_calls
            ai_message.content = ""  # Vaciamos para que LangGraph ejecute las herramientas
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