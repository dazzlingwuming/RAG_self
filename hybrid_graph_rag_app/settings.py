from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CONFIG_DIR = PROJECT_ROOT / "config"
DOTENV_PATH = CONFIG_DIR / ".env"

MODEL_DIR = PROJECT_ROOT / "model"
EMBEDDING_MODEL_PATH = MODEL_DIR / "bge-base-zh-v1.5"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore_rag"

DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
HISTORY_PATH = DATA_DIR / "conversation_history.json"

NEO4J_HOME = PROJECT_ROOT / "neo4j_kg_db"
NEO4J_JAVA_HOME = (
    Path.home()
    / ".Neo4jDesktop2"
    / "Cache"
    / "runtime"
    / "zulu21.44.17-ca-jdk21.0.8-win_x64"
)
NEO4J_BOLT_URL = "bolt://127.0.0.1:8687"
NEO4J_HTTP_URL = "http://127.0.0.1:8474"
NEO4J_START_SCRIPT = NEO4J_HOME / "start_kg_db.bat"
NEO4J_STOP_SCRIPT = NEO4J_HOME / "stop_kg_db.bat"

VECTOR_TOP_K = 6
GRAPH_KEYWORD_LIMIT = 3
GRAPH_RESULT_LIMIT = 8

