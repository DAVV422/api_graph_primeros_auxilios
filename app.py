from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from langchain.memory import ConversationBufferMemory # Mantener si se planea usar LLM
from neo4j import AsyncGraphDatabase
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import re
from datetime import datetime, timedelta
import google.generativeai as genai # Nuevo: Importar Gemini API
import os # Nuevo: Para cargar variables de entorno
from dotenv import load_dotenv # Nuevo: Para cargar .env file

# Cargar variables de entorno al inicio
load_dotenv()

# --- Configuración inicial ---
app = FastAPI()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === INSERTA EL CÓDIGO DE CONFIGURACIÓN DE CORS AQUÍ ===
# Configuración de CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # O especificar ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"], # O especificar headers específicos
)

# 1. Conexión a Neo4j
# Asegúrate de que Neo4j esté corriendo y las credenciales sean correctas
#driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neoia123.")) # Reemplaza con tus credenciales
#driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
neo4j_driver = None

# 2. Memoria global para el estado de las sesiones
# Aquí se guarda el estado de la conversación para cada usuario.
session_states: Dict[str, Dict[str, Any]] = {}  # {"session_id": {emergency, last_node_id, last_node_type, waiting_for_answer, last_interaction_time}}

# 3. Tareas de timeout para limpiar sesiones inactivas
timeout_tasks: Dict[str, asyncio.Task] = {}

# 4. LangChain Memory (Opcional, solo si usas LLM para generación de respuestas)
# Se inicializa solo si se va a usar para mantener el contexto de conversación más allá del grafo.
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# --- Eventos de ciclo de vida de FastAPI para el driver de Neo4j ---
@app.on_event("startup")
async def startup_event():
    global neo4j_driver # Usa la variable global
    try:
        # Inicializa el driver asíncrono
        neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        # Verifica la conectividad (opcional pero recomendado)
        await neo4j_driver.verify_connectivity()
        print("Conexión con Neo4j Aura establecida correctamente (asíncrona).")
    except Exception as e:
        print(f"ERROR: Se produjo un error inesperado al conectar a Neo4j: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado al conectar a Neo4j: {e}")


print()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# --- Funciones de interacción con Neo4j ---

async def get_next_interaction(
    session_id: str,
    emergency_name: str,
    user_input_text: str = None, # La respuesta del usuario
    last_node_id: int = None,    # El ID del nodo anterior (Evaluacion o Paso)
    last_node_type: str = None,  # El tipo de nodo anterior ('question' o 'step')
    is_waiting_for_answer: bool = False # Indica si se está esperando un 'SI'/'NO'
) -> Dict[str, Any]:
    """
    Recupera la siguiente pregunta o paso de la guía desde Neo4j
    basándose en el estado actual de la conversación y la entrada del usuario.
    """
    with neo4j_driver.session() as session:
        # Escenario 1: Es la primera interacción para una emergencia o se reinicia el flujo
        if last_node_id is None and not is_waiting_for_answer:
            # Busca la primera evaluación (pregunta) para la emergencia
            query = """
            MATCH (e:Emergencia {nombre: $name})-[:TIENE_EVALUACION]->(ev)
            RETURN ev.pregunta AS content, ID(ev) AS node_id, 'question' AS type
            ORDER BY ID(ev) LIMIT 1
            """
            result = session.run(query, name=emergency_name)
            record = result.single()
            if record:
                return {
                    "type": record["type"],
                    "content": record["content"],
                    "node_id": record["node_id"],
                    "is_end": False # No es el final del flujo
                }
            else:
                return {"type": "end", "content": "No se encontraron evaluaciones para esta emergencia. Por favor, busque ayuda médica.", "is_end": True}

        # Escenario 2: El usuario está respondiendo a una pregunta (esperando 'SI' o 'NO')
        elif is_waiting_for_answer and last_node_type == 'question':
            # Determina si la respuesta del usuario es 'SI' o 'NO'
            response_lower = user_input_text.lower().strip()
            rel_type = None # Tipo de relación a buscar ('SI' o 'NO')

            if "si" in response_lower or "sí" in response_lower or "yes" in response_lower:
                rel_type = "SI"
            elif "no" in response_lower or "not" in response_lower:
                rel_type = "NO"

            if rel_type:
                # Busca el primer paso que sigue a la evaluación con la respuesta dada
                query = f"""
                MATCH (ev:Evaluacion)
                WHERE ID(ev) = $last_node_id
                MATCH (ev)-[rel:{rel_type}]->(paso:Paso)
                RETURN paso.accion AS content, ID(paso) AS node_id, 'step' AS type
                ORDER BY paso.orden LIMIT 1
                """
                result = session.run(query, last_node_id=last_node_id)
                record = result.single()
                if record:
                    return {
                        "type": record["type"],
                        "content": record["content"],
                        "node_id": record["node_id"],
                        "is_end": False
                    }
                else:
                    # Esto podría ocurrir si no hay un paso directo para esa respuesta
                    return {"type": "end", "content": "No se encontró el paso siguiente para su respuesta. Por favor, busque ayuda médica.", "is_end": True}
            else:
                # Si la respuesta no es clara, se pide al usuario que aclare
                return {"type": "clarify", "content": "Por favor, responda 'sí' o 'no' para continuar.", "is_end": False}

        # Escenario 3: Continuar la secuencia de pasos (después de haber recibido un paso)
        elif last_node_type == 'step' and not is_waiting_for_answer:
            # Busca el siguiente paso en la secuencia (relación SIGUE)
            query = """
            MATCH (current_paso:Paso)
            WHERE ID(current_paso) = $last_node_id
            OPTIONAL MATCH (current_paso)-[:SIGUE]->(next_paso:Paso)
            RETURN next_paso.accion AS content, ID(next_paso) AS node_id, 'step' AS type
            ORDER BY next_paso.orden LIMIT 1
            """
            result = session.run(query, last_node_id=last_node_id)
            record = result.single()

            if record and record["content"]: # Asegurarse de que se encontró un siguiente paso
                return {
                    "type": record["type"],
                    "content": record["content"],
                    "node_id": record["node_id"],
                    "is_end": False
                }
            else:
                # No hay más pasos en esta rama
                return {"type": "end", "content": "Hemos llegado al final de los pasos para esta rama. Por favor, busque ayuda médica si es necesario.", "is_end": True}

        # Fallback para estados inesperados
        return {"type": "error", "content": "Ocurrió un error inesperado en el flujo de la conversación.", "is_end": True}


async def classify_with_gemini(text: str, session_id: str = "default") -> str:
    """
    Clasifica el texto de entrada usando Gemini según las categorías de emergencia definidas.
    - Si encuentra una coincidencia clara, devuelve el nombre formal de la emergencia.
    - Si no aplica, devuelve el texto original sin modificar.

    Args:
        text: Texto del usuario a clasificar
        session_id: ID para mantener contexto de conversación (opcional)

    Returns:
        str: Nombre de la emergencia o texto original
    """
    # Lista completa de emergencias (actualizable)
    EMERGENCIAS_VALIDAS = [
        "Cuerpo Extraño en el Ojo",
        "Quemaduras de Segundo Grado",
        "Quemaduras Eléctricas",
        "Atragantamiento en Adultos y Niños Mayores",
        "Convulsiones (Post-Convulsión y Protección)",
        "Dientes Rotos o Caídos",
        "Dificultad para Respirar (Leve)",
        "Fracturas Evidentes o Sospechosas",
        "Hemorragia Severa",
        "Ahogamiento",
        "Golpe en la Cabeza",
        "Cortes y Raspaduras Menores",
        "Esguinces y Torceduras Leves",
        "Picaduras de Insectos (No Alérgicas)",
        "Golpes y Contusiones Menores",
        "Sangrado Nasal",
        "Insolación Leve / Agotamiento por Calor",
        "Hipotermia Leve",
        "Desmayo (Síncope Simple)",
        "RCP en Niños (1 a 8 años)",
        "RCP en Bebés (< 1 año)",
        "RCP en Adultos (Solo Manos)",
        "Revisión Básica de Conciencia y Respiración"
    ]

    # Prompt optimizado para clasificación estricta
    CLASSIFICATION_PROMPT = f"""
    Eres un clasificador médico de emergencias. Analiza el texto del usuario y:
    1. Devuelve **EXACTAMENTE** uno de estos nombres de emergencia si hay coincidencia CLARA:
    {EMERGENCIAS_VALIDAS}

    2. Si el texto NO describe una emergencia médica o es ambiguo, devuelve **el texto original tal cual**.

    Reglas:
    - No agregues explicaciones.
    - No modifiques el texto original si no es una emergencia.
    - Usa solo los nombres de emergencia proporcionados.

    Texto a clasificar: "{text}"
    """

    try:
        # Llamada a Gemini
        response = await _generate_llm_response(
            raw_content=CLASSIFICATION_PROMPT,
            session_id=session_id,
            original_type="classification"
        )

        # Verificación estricta de la respuesta
        if response in EMERGENCIAS_VALIDAS:
            return response
        else:
            return text

    except Exception as e:
        print(f"Error en clasificación con Gemini: {e}")
        return text  # Fallback seguro
    
    

async def identify_emergency(text: str) -> str:
    """
    Identifica la emergencia principal basada en palabras clave del texto.
    Esta función podría ser mejorada con un LLM para mayor precisión.
    """
    emergency_keywords = {
        "ojo": "Cuerpo Extraño en el Ojo",
        "quemadura": "Quemaduras de Segundo Grado",
        "electrica": "Quemaduras Eléctricas",
        "atragantamiento": "Atragantamiento en Adultos y Niños Mayores",
        "convulsion": "Convulsiones (Post-Convulsión y Protección)",
        "diente": "Dientes Rotos o Caídos",
        "respirar": "Dificultad para Respirar (Leve)",
        "fractura": "Fracturas Evidentes o Sospechosas",
        "hemorragia": "Hemorragia Severa",
        "ahogamiento": "Ahogamiento",
        "cabeza": "Golpe en la Cabeza",
        "corte": "Cortes y Raspaduras Menores",
        "esguince": "Esguinces y Torceduras Leves",
        "picadura": "Picaduras de Insectos (No Alérgicas)",
        "golpe": "Golpes y Contusiones Menores",
        "sangrado nasal": "Sangrado Nasal",
        "insolacion": "Insolación Leve / Agotamiento por Calor",
        "hipotermia": "Hipotermia Leve",
        "desmayo": "Desmayo (Síncope Simple)",
        "rcp niños": "RCP en Niños (1 a 8 años)",
        "rcp bebes": "RCP en Bebés (< 1 año)",
        "rcp": "RCP en Adultos (Solo Manos)", # General RCP, order matters here
        "revision": "Revisión Básica de Conciencia y Respiración"
    }

    text_lower = text.lower()
    #text_lower = await classify_with_gemini(text_lower)
    # Priorizar coincidencias más específicas o más largas si hay solapamiento
    # Por ejemplo, "rcp bebes" antes que "rcp"
    sorted_keywords = sorted(emergency_keywords.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in text_lower:
            return emergency_keywords[keyword]

    return "No se pudo identificar la emergencia específica. Por favor, describa con más detalle o diga 'ayuda' para obtener un listado de emergencias."


async def reset_if_no_response(session_id: str):
    """
    Función para resetear el estado de la sesión si no hay actividad
    después de un período de tiempo.
    """
    await asyncio.sleep(60) # Espera 60 segundos
    if session_id in session_states:
        # Verifica que la última interacción haya sido hace al menos 55 segundos
        # para evitar un reset si una respuesta llegó justo al final del sleep.
        if datetime.now() - session_states[session_id]["last_interaction_time"] >= timedelta(seconds=55):
            print(f"Session {session_id} timed out. Resetting state.")
            session_states[session_id] = {
                "emergency": None,
                "last_node_id": None,
                "last_node_type": None,
                "waiting_for_answer": False,
                "last_interaction_time": datetime.now() # Actualiza el tiempo de reseteo
            }
            if session_id in conversation_memories:
                del conversation_memories[session_id]
            # Opcional: Aquí podrías enviar un mensaje automático al usuario
            # indicando que la sesión ha expirado o que está esperando una nueva consulta.
            # (Esto requeriría un mecanismo de envío de mensajes asíncrono hacia el cliente)
            print(f"Bot (a {session_id}): Parece que no hay actividad. ¿Hay algo más en lo que pueda ayudarte?")


async def _generate_llm_response(raw_content: str, session_id: str, is_emergency_end: bool = False, original_type: str = "") -> str:
    """
    Genera una respuesta suavizada y amigable usando Gemini,
    adaptando el tono para emergencias y sugiriendo una ambulancia si es el final del flujo.
    """
    
    memory = conversation_memories.get(session_id)
    if not memory:
        memory = ConversationBufferMemory()
        conversation_memories[session_id] = memory

    system_instruction = (
        "Eres un asistente de primeros auxilios. Tu objetivo es comunicar el paso o pregunta que te proporcionaré de manera "
        "extremadamente CLARA, SENCILLA y CALMADA. Usa un tono AMABLE y RECONFORTANTE. "
        "Cada respuesta debe ser breve y al punto, como si estuvieras guiando a alguien en una situación de estrés."
        "Si te paso el siguiente texto como paso o pregunta: ''No se pudo identificar la emergencia específica. Por favor, describa con más detalle o diga 'ayuda' para obtener un listado de emergencias.'' "
        "debes responder: que la emergencia está fuera de tu alcance y que se llame a la ambulancia lo más pronto posible, importante: No le pidas datos en esa situación. "
    )

    prompt_text = raw_content

    print(raw_content)
    if original_type == "question":
        prompt_text = f"La pregunta es: '{raw_content}'. Por favor, formúlala en un tono calmado y amigable."
    elif original_type == "step":
        prompt_text = f"El paso a seguir es: '{raw_content}'. Por favor, explícalo de forma calmada y sencilla."
    
    if is_emergency_end:
        # Si el flujo ha terminado o hay una ambigüedad en el grafo.
        # Prioridad absoluta: recomendar ambulancia.
        system_instruction += (
            "\nIMPORTANTE: Si el flujo de primeros auxilios ha terminado o no hay una guía clara, "
            "tu prioridad es indicar al usuario que llame a una ambulancia (160 en Bolivia) INMEDIATAMENTE. "
            "Asegúrate de que este mensaje sea el principal y muy claro, reforzando la urgencia. "
            "No extiendas el mensaje con otra información, solo la llamada a emergencias."
        )
        prompt_text = f"La guía no puede continuar o ha concluido. Mensaje del sistema: '{raw_content}'. Debes indicar que llamen a emergencias."


    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=system_instruction,            
        )
        # Convertir historial de LangChain a formato de Gemini
        # Esto es un ejemplo, adapta según la estructura real de tu LangChain Memory
        langchain_history = memory.load_memory_variables({})['history'] if 'history' in memory.load_memory_variables({}) else []
        gemini_chat_history = []
        for entry in langchain_history:
            if isinstance(entry, dict) and 'role' in entry and 'parts' in entry: # Check if it's already in Gemini format
                gemini_chat_history.append(entry)
            elif isinstance(entry, tuple) and len(entry) == 2: # Assuming (user_msg, bot_msg)
                gemini_chat_history.append({'role': 'user', 'parts': [{'text': entry[0]}]})
                gemini_chat_history.append({'role': 'model', 'parts': [{'text': entry[1]}]})
            # Handle other formats if your LangChain memory stores them differently
            # For ConversationBufferMemory, it often stores as a list of HumanMessage and AIMessage objects
            elif hasattr(entry, 'type'):
                if entry.type == 'human':
                    gemini_chat_history.append({'role': 'user', 'parts': [{'text': entry.content}]})
                elif entry.type == 'ai':
                    gemini_chat_history.append({'role': 'model', 'parts': [{'text': entry.content}]})


        convo = model.start_chat(history=gemini_chat_history)
        response = convo.send_message(prompt_text)
        return response.text
    except Exception as e:
        print(f"Error al llamar a la API de Gemini: {e}")
        # Fallback a un mensaje genérico o al contenido original
        if is_emergency_end:
            return "Lo siento, no puedo procesar la información en este momento. Por favor, ¡comuníquese con una ambulancia al 160 lo antes posible!"
        return f"Lo siento, tuve un problema al generar la respuesta. El mensaje original era: {raw_content}. Por favor, busque ayuda médica si es necesario."


# --- Modelos Pydantic para los endpoints ---
class UserMessage(BaseModel):
    text: str
    session_id: str

# --- API Endpoint ---
@app.post("/chat")
async def chat(message: UserMessage):
    # 1. Inicializar el estado de la sesión si es nueva
    if message.session_id not in session_states:
        session_states[message.session_id] = {
            "emergency": None,
            "last_node_id": None,
            "last_node_type": None,
            "waiting_for_answer": False,
            "last_interaction_time": datetime.now()
        }
        conversation_memories[message.session_id] = ConversationBufferMemory() # Inicializar LangChain memory

    state = session_states[message.session_id]
    state["last_interaction_time"] = datetime.now() # Actualizar tiempo de última interacción

    # 2. Cancelar la tarea de timeout previa para esta sesión
    if message.session_id in timeout_tasks:
        timeout_tasks[message.session_id].cancel()
        del timeout_tasks[message.session_id]

    raw_response_content = "" # Contenido sin procesar del grafo
    response_type = "" # Tipo de respuesta del grafo (question, step, end, clarify)
    is_flow_end = False # Indicador si el flujo del grafo ha terminado

    user_text_lower = message.text.lower().strip()

    # Comandos para controlar el flujo
    if user_text_lower in ["iniciar", "empezar", "reset", "reiniciar", "comenzar"]:
        # Reiniciar completamente la sesión
        state["emergency"] = None
        state["last_node_id"] = None
        state["last_node_type"] = None
        state["waiting_for_answer"] = False
        raw_response_content = "Hola, soy tu guía de primeros auxilios. Por favor, describe brevemente la emergencia para comenzar (ej: 'Me corté un dedo', 'Alguien se atraganta')."
        response_type = "initial_message" # Nuevo tipo para mensajes de inicio        
    elif user_text_lower == "siguiente paso" and state["last_node_type"] == "step" and not state["waiting_for_answer"]:
        # Si el usuario pide explícitamente el siguiente paso y el último fue un paso
        next_interaction = await get_next_interaction(
            session_id=message.session_id,
            emergency_name=state["emergency"],
            last_node_id=state["last_node_id"],
            last_node_type=state["last_node_type"],
            is_waiting_for_answer=False
        )
        raw_response_content = next_interaction["content"]
        response_type = next_interaction["type"]
        state["last_node_id"] = next_interaction["node_id"]
        state["last_node_type"] = next_interaction["type"]
        state["waiting_for_answer"] = (next_interaction["type"] == "question")
        is_flow_end = next_interaction["is_end"]
    elif state["emergency"] is None or state["last_node_type"] == "end":
        # Identificar la emergencia si no se ha hecho o si el flujo anterior terminó
        emergency_identified = await identify_emergency(message.text)
        if emergency_identified == "No se pudo identificar la emergencia específica. Por favor, describa con más detalle o diga 'ayuda' para obtener un listado de emergencias.":
            raw_response_content = emergency_identified
            response_type = "clarify" # Tipo para mensajes de aclaración
            state["emergency"] = None # Asegurar que la emergencia no esté establecida si no se identificó
            state["last_node_type"] = None
            state["last_node_id"] = None
            state["waiting_for_answer"] = False
        else:
            state["emergency"] = emergency_identified
            state["last_node_id"] = None # Reiniciar para el flujo de la nueva emergencia
            state["last_node_type"] = None
            state["waiting_for_answer"] = False # Asegurarse de que no esté esperando respuesta para la pregunta inicial

            next_interaction = await get_next_interaction(
                session_id=message.session_id,
                emergency_name=state["emergency"],
                # Para la primera llamada, last_node_id y last_node_type son None
            )
            raw_response_content = next_interaction["content"]
            response_type = next_interaction["type"]
            state["last_node_id"] = next_interaction["node_id"]
            state["last_node_type"] = next_interaction["type"]
            state["waiting_for_answer"] = (next_interaction["type"] == "question")
            is_flow_end = next_interaction["is_end"]
    else:
        # Continuar el flujo de la emergencia actual
        next_interaction = await get_next_interaction(
            session_id=message.session_id,
            emergency_name=state["emergency"],
            user_input_text=message.text, # Pasa el texto del usuario para las respuestas SI/NO
            last_node_id=state["last_node_id"],
            last_node_type=state["last_node_type"],
            is_waiting_for_answer=state["waiting_for_answer"]
        )

        raw_response_content = next_interaction["content"]
        response_type = next_interaction["type"]
        state["last_node_id"] = next_interaction["node_id"]
        state["last_node_type"] = next_interaction["type"]
        state["waiting_for_answer"] = (next_interaction["type"] == "question") # Solo se espera respuesta si lo siguiente es una pregunta
        is_flow_end = next_interaction["is_end"]

        # Manejar el caso de que el flujo haya terminado
        if is_flow_end:
            state["emergency"] = None # Resetear la emergencia
            state["last_node_type"] = "end" # Marcar como finalizado
    
    # Procesar la respuesta con Gemini para suavizarla
    final_response_content = await _generate_llm_response(
        raw_response_content,
        message.session_id,
        is_emergency_end=is_flow_end,
        original_type=response_type
    )

    # Si se está esperando una respuesta (pregunta activa), se programa la tarea de timeout
    if state["waiting_for_answer"]:
        timeout_tasks[message.session_id] = asyncio.create_task(reset_if_no_response(message.session_id))

    return {"response": final_response_content}

# Endpoint para cerrar el driver de Neo4j al cerrar la aplicación
@app.on_event("shutdown")
async def shutdown_event():
    neo4j_driver.close()
    print("Neo4j driver closed.")
