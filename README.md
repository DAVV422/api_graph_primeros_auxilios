# 🩺 Asistente de Primeros Auxilios con FastAPI, Neo4j y Gemini

Este proyecto proporciona una API inteligente para asistir a personas sin conocimientos médicos en **emergencias leves**, guiándolas paso a paso mediante preguntas y acciones específicas, según el tipo de emergencia. Utiliza:

- 🧠 **Neo4j** como grafo de conocimiento estructurado.
- 🤖 **Google Gemini (LLM)** para mejorar comprensión y generación de respuestas suaves.
- ⚙️ **FastAPI** para la interfaz web asíncrona.
- 🧵 **LangChain Memory** para mantener el contexto conversacional.
- 🔐 `.env` para configuración segura.

---

## 🚀 Características

- Clasificación automática de emergencias médicas leves usando Gemini.
- Flujo dinámico de pasos y preguntas extraídos desde un grafo Neo4j.
- Comunicación conversacional clara y reconfortante.
- Soporte para sesiones múltiples con memoria.
- Reinicio automático de sesión por inactividad.
- API lista para frontend, móvil o asistentes por voz.

---

## 📦 Tecnologías Utilizadas

- Python 3.10+
- FastAPI
- Neo4j (Driver asíncrono)
- LangChain (opcional)
- Google Generative AI (Gemini 2.0 Flash)
- dotenv (manejo de variables de entorno)
- CORS Middleware para pruebas o conexión con frontend

---

## 🔧 Instalación

1. **Clonar el repositorio:**
```bash
https://github.com/DAVV422/api_graph_primeros_auxilios.git
```

2. **Crea y activa un entorno virtual (opcional):**
```
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instala las dependencias:**
```
pip install -r requirements.txt
```

4. **Crea un archivo `.env` con tus claves de acceso:**
```
# .env
NEO4J_URI=bolt+s://<tu-servidor-neo4j>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=tu_contraseña
GOOGLE_API_KEY=tu_api_key_de_gemini
```

## 3. 🧠 Estructura del Grafo de Conocimiento

El grafo está construido en **Neo4j**, utilizando **Cypher** para definir la lógica y conexiones entre nodos.

En el repositorio, la carpeta `data_knowledge_graph/` contiene:

- `grafo.png` (o nombre similar): imagen visual del grafo exportada desde Neo4j Bloom.
- `bloom-export.zip`: archivo exportado desde Bloom que incluye los nodos y relaciones en hojas de cálculo Excel.

### 🔹 Modelo de grafo:

```
(:Emergencia)-[:TIENE_EVALUACION]->(:Evaluacion)
(:Evaluacion)-[:SI|NO]->(:Paso)
(:Paso)-[:SIGUE]->(:Paso)
```

- `Emergencia`: tipo raíz (ej. “Atragantamiento”)
- `Evaluacion`: pregunta de decisión (ej. “¿Está respirando?”)
- `Paso`: acción específica a realizar (ej. “Presione el pecho con fuerza”)

``### 🛠 Cómo importar el grafo (opcional)

1. Descomprime `bloom-export.zip`.
2. Usa Neo4j Desktop o Neo4j Aura para importar:
   - Opción 1: con el **Data Importer** de Neo4j.
   - Opción 2: con comandos Cypher como este:

``
CREATE (e:Emergencia {nombre: "Insolación"})
CREATE (q:Evaluacion {pregunta: "¿Está consciente?"})
CREATE (p1:Paso {accion: "Llévalo a la sombra y dale agua en sorbos"})
CREATE (e)-[:TIENE_EVALUACION]->(q)
CREATE (q)-[:SI]->(p1)
``

Puedes construir secuencias encadenadas usando la relación `SIGUE`.

## 4. 🧪 Ejecución local

```
uvicorn main:app --reload
```

Visita en tu navegador:

```
http://localhost:8000/docs
```

## 5. 📡 Endpoint principal

### POST /chat

```
{
  "text": "alguien se atraganta",
  "session_id": "usuario123"
}
```

Respuesta esperada:

```
{
  "response": "Mantén la calma. ¿La persona puede toser o hablar?"
}
```

## 6. ⏱ Gestión de Sesiones

- Cada usuario tiene un `session_id` que mantiene su contexto.
- Las sesiones se reinician automáticamente tras 60 segundos sin interacción.
- Se puede forzar reinicio con: `iniciar`, `reiniciar`, `reset`.

## 7. 🤖 Gemini y LangChain

- Se utiliza **Gemini 2.0 Flash** para convertir pasos y preguntas en mensajes claros y calmados.
- **LangChain** (`ConversationBufferMemory`) almacena el contexto reciente de la conversación.

## 8. ⚙️ Comandos especiales

- `"iniciar"` o `"reset"`: reinicia la sesión.
- `"siguiente paso"`: fuerza avanzar al próximo paso si el flujo lo permite.
- `"ayuda"`: muestra sugerencias cuando no se identifica una emergencia.

## 9. 📦 Despliegue

Puedes desplegar este backend en:

- **Railway, Render, Fly.io**
- **VPS** (DigitalOcean, Contabo, etc.)
- **Docker** (puedes agregar un `Dockerfile` personalizado)

## 10. 📁 Estructura del Proyecto

```
├── main.py                # Código principal del backend FastAPI
├── data_knowledge_graph/  # Grafo exportado: imágenes y archivos Bloom
├── requirements.txt       # Lista de dependencias
└── .env                   # Variables de entorno
```

## 11. 🛡️ Licencia

MIT © 2025 - Diego A. Vargas Vaca

## 12. 📬 Contacto

- GitHub: [DAVV422](https://github.com/DAVV422)
- Email: diegoalberto42216@gmail.com
- LinkedIn: [www.linkedin.com/in/davv42216](https://www.linkedin.com/in/davv42216)
