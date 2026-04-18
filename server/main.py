import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ragu import (
    KnowledgeGraph,
    SimpleChunker,
    BuilderArguments,
    LocalSearchEngine,
    GlobalSearchEngine,
    NaiveSearchEngine,
    Settings,
)
from ragu.models import LLMOpenAI
from ragu.models.openai import CachedAsyncOpenAI
from ragu.triplet import ArtifactsExtractorLLM
from etl.local_embedder import LocalEmbedder


ENV_KEYS = [
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_RPM",
    "EMBEDDER_API_KEY", "EMBEDDER_BASE_URL", "EMBEDDER_MODEL", "EMBEDDER_DIM",
    "RAGU_STORAGE",
]


class State:
    kg: KnowledgeGraph | None = None
    llm: LLMOpenAI | None = None
    embedder: LocalEmbedder | None = None
    indexed: bool = False
    all_documents: list[str] = []
    all_names: list[str] = []


state = State()


def reinit_clients():
    api_key = os.getenv("LLM_API_KEY_RUNTIME", os.getenv("LLM_API_KEY", ""))

    client = CachedAsyncOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://models.github.ai/inference"),
        api_key=api_key,
        rate_max_per_minute=10,
        retry_times_sec=(15, 30, 60),
    )
    state.llm = LLMOpenAI(
        client=client,
        model_name=os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"),
    )

    state.embedder = LocalEmbedder("intfloat/multilingual-e5-small")


@asynccontextmanager
async def lifespan(app: FastAPI):
    reinit_clients()

    # Auto-load pre-built index if exists
    storage = os.getenv("RAGU_STORAGE", "ragu_data")
    chunks_path = os.path.join(storage, "kv_chunks.json")
    if os.path.exists(chunks_path):
        try:
            Settings.storage_folder = storage
            state.kg = KnowledgeGraph(
                llm=state.llm,
                embedder=state.embedder,
            )
            state.indexed = True
            print(f"[RAGU] Auto-loaded index from {storage}")
        except Exception as e:
            print(f"[RAGU] Failed to auto-load index: {e}")

    yield


app = FastAPI(title="RAGU API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    engine: str = "local"
    top_k: int = 20


class BotResponse(BaseModel):
    message: str
    files: list[str] = []
    document: str | None = None
    confidence: float = 0.0
    digest: str = ""
    title: str = ""


class ConfigUpdate(BaseModel):
    env: dict[str, str]


@app.get("/api/status")
async def get_status():
    return {"indexed": state.indexed}


@app.get("/api/config")
async def get_config():
    return {k: os.getenv(k, "") for k in ENV_KEYS}


@app.post("/api/config")
async def set_config(req: ConfigUpdate):
    for k, v in req.env.items():
        if k in ENV_KEYS:
            os.environ[k] = v
    reinit_clients()
    return {k: os.getenv(k, "") for k in ENV_KEYS}


async def _read_file(upload: UploadFile) -> str:
    content = await upload.read()
    ext = (upload.filename or "").rsplit(".", 1)[-1].lower()

    if ext == "docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    return content.decode("utf-8")


@app.post("/api/index")
async def index_documents(
    files: list[UploadFile] = File(default=[]),
    text: str = Form(default=""),
    language: str = Form(default="russian"),
):
    documents: list[str] = []
    names: list[str] = []

    if text.strip():
        documents.append(text.strip())
        names.append("(text)")

    for f in files:
        doc_text = await _read_file(f)
        if doc_text.strip():
            documents.append(doc_text)
            names.append(f.filename or "unknown")

    if not documents:
        return {"status": "empty", "documents_count": 0, "names": []}

    state.all_documents.extend(documents)
    state.all_names.extend(names)

    try:
        Settings.storage_folder = os.getenv("RAGU_STORAGE", "ragu_data")
        Settings.language = language

        chunker = SimpleChunker(max_chunk_size=1000)
        extractor = ArtifactsExtractorLLM(llm=state.llm, do_validation=False)

        kg = KnowledgeGraph(
            llm=state.llm,
            embedder=state.embedder,
            chunker=chunker,
            artifact_extractor=extractor,
            builder_settings=BuilderArguments(
                use_llm_summarization=True,
                vectorize_chunks=True,
            ),
        )

        await kg.build_from_docs(state.all_documents)

        state.kg = kg
        state.indexed = True

    except Exception as e:
        # Roll back documents added in this request
        state.all_documents = state.all_documents[:-len(documents)]
        state.all_names = state.all_names[:-len(names)]
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "names": names, "documents_count": 0},
        )

    return {
        "status": "indexed",
        "documents_count": len(documents),
        "names": names,
        "total_documents": len(state.all_documents),
    }


@app.post("/api/query", response_model=BotResponse)
async def query_graph(req: QueryRequest):
    if not state.kg:
        return BotResponse(
            message="Knowledge graph not built yet. Please index documents first.",
        )

    try:
        if req.engine == "local":
            engine = LocalSearchEngine(state.llm, state.kg, state.embedder)
            answer = await engine.a_query(req.query, top_k=req.top_k)
        elif req.engine == "global":
            engine = GlobalSearchEngine(state.llm, state.kg)
            answer = await engine.a_query(req.query)
        elif req.engine == "naive":
            engine = NaiveSearchEngine(state.llm, state.kg, state.embedder)
            answer = await engine.a_query(req.query, top_k=req.top_k)
        else:
            return BotResponse(message=f"Unknown engine: {req.engine}")
    except Exception as e:
        return BotResponse(message=f"Error: {e}")

    # a_query() may return a string or a Pydantic model with .response
    text = answer.response if hasattr(answer, "response") else str(answer)

    return BotResponse(
        message=text,
        confidence=0.8,
        title=req.query[:50],
    )
