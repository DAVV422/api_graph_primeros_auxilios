from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any
import asyncio
from datetime import datetime, timedelta

# --- Configuración inicial ---
app = FastAPI()

# 1. Conexión a Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neoia123."))


# 2. Memoria global (por sesión)
session_states: Dict[str, Dict[str, Any]] = {}  # {"session_id": {estados}}
timeout_tasks: Dict[str, asyncio.Task] = {}  # Para manejar timeouts

# 3. LangChain Memory (Opcional, solo si usas LLM para generación de respuestas)
conversation_memories: Dict[str, ConversationBufferMemory] = {}

# --- Funciones del sistema ---
def get_next_step(emergency_name: str, step_number: int, user_response: str = None) -> Dict[str, str]:
    with driver.session() as session:
        # Consulta evaluaciones primero
        if step_number == 1:
            query = """
            MATCH (e:Emergencia {nombre: $name})-[:TIENE_EVALUACION]->(ev)
            RETURN ev.pregunta AS pregunta
            ORDER BY ID(ev) LIMIT 1
            """
            result = session.run(query, name=emergency_name)
            record = result.single()
            if record:
                return {"type": "question", "content": record["pregunta"]}
        
        # Consulta pasos basados en respuesta previa
        query = """
        MATCH (e:Emergencia {nombre: $name})-[:TIENE_EVALUACION]->(ev)
        WHERE ev.pregunta = $last_question
        MATCH (ev)-[:SI|NO]->(paso)
        WHERE paso.orden = $step
        RETURN paso.accion AS accion
        """
        result = session.run(query, name=emergency_name, 
                           last_question=user_response, 
                           step=step_number)
        record = result.single()
        
        return {"type": "step", "content": record["accion"] if record else "Por favor busque ayuda médica"}

def identify_emergency(text: str) -> str:
    """Identifica la emergencia usando palabras clave del grafo"""
    emergency_keywords = {
        "ojo": "Cuerpo Extraño en el Ojo",
        "quem": "Quemaduras de Segundo Grado",
        "electri": "Quemaduras Eléctricas",
        "atragant": "Atragantamiento en Adultos y Niños Mayores",
        "convul": "Convulsiones (Post-Convulsión y Protección)"
    }
    
    text_lower = text.lower()
    for keyword, emergency in emergency_keywords.items():
        if keyword in text_lower:
            return emergency
    return "Cortes y Raspaduras Menores"  # Default seguro

async def reset_if_no_response(session_id: str):
    """Cancela la espera después de 30 segundos si no hay respuesta."""
    await asyncio.sleep(30)
    if session_id in session_states and session_states[session_id]["waiting_for_answer"]:
        session_states[session_id]["waiting_for_answer"] = False
        # Opcional: Enviar un mensaje automático
        # Ej: "Continuamos con el siguiente paso: [acción]"

# --- API Endpoint ---
class UserMessage(BaseModel):
    text: str
    session_id: str

@app.post("/chat")
async def chat(message: UserMessage):
    # 1. Inicializar sesión si es nueva
    if message.session_id not in session_states:
        session_states[message.session_id] = {
            "emergency": None,
            "current_step": 1,
            "waiting_for_answer": False,
            "last_interaction": datetime.now()
        }
        conversation_memories[message.session_id] = ConversationBufferMemory()  # Opcional
    
    state = session_states[message.session_id]
    state["last_interaction"] = datetime.now()

    # 2. Cancelar timeout previo si existe
    if message.session_id in timeout_tasks:
        timeout_tasks[message.session_id].cancel()

    # 3. Determinar si es respuesta a pregunta previa
    if state["waiting_for_answer"]:
        response = get_next_step(state["emergency"], state["current_step"], message.text)
        state["waiting_for_answer"] = False
        state["current_step"] += 1
    else:
        # 4. Nueva emergencia
        emergency = identify_emergency(message.text)
        state["emergency"] = emergency
        state["current_step"] = 1
        response = get_next_step(emergency, 1)

    # 5. Manejar preguntas (activar timeout)
    if response["type"] == "question":
        state["waiting_for_answer"] = True
        timeout_tasks[message.session_id] = asyncio.create_task(reset_if_no_response(message.session_id))

    # 6. Opcional: Usar LLM para enriquecer respuestas (si conversation_memories está activo)
    #memory = conversation_memories.get(message.session_id)
    #print(f"memory: ${memory}")
    #if memory:
        #memory.save_context({"input": message.text}, {"output": response["content"]})
        # Ejemplo de enriquecimiento con LLM:
        # response["content"] = llm.generate(...)

    return {"response": response["content"]}