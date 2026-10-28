"""
FastAPI prediction service — production grade.
Run with: uvicorn api.main:app --reload

Features:
- Rate limiting (10 req/min per IP via slowapi)
- Multi-ticker support
- Model versioning info
- /health, /predict, /features, /model_info, /tickers endpoints
"""
from __future__ import annotations
import os
import sys
import json
import logging
import asyncio
import time
from functools import wraps
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

try:
    from sse_starlette.sse import EventSourceResponse
except ImportError:
    EventSourceResponse = None

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer("alpha-engine.api")
except ImportError:
    trace = None
    _tracer = None

try:
    from jose import JWTError, jwt
except ImportError:
    JWTError = Exception
    jwt = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

from src.modeling import FEATURES
from src.feature_utils import compute_features_from_ohlcv
from src.streaming.kafka_producer import InteractionEvent, InteractionProducer

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Alpha Engine API",
    description="Multi-ticker stock return prediction using stacking ensemble.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=os.getenv("OAUTH2_TOKEN_URL", "/oauth/token"))
_redis = None
_interaction_producer: InteractionProducer | None = None
_interaction_producer_lock = asyncio.Lock()


def _span(name: str):
    """Return an OpenTelemetry span, or a no-op context if telemetry is absent."""
    if _tracer is not None:
        return _tracer.start_as_current_span(name)
    from contextlib import nullcontext
    return nullcontext()


def traced(name: str):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with _span(name):
                return func(*args, **kwargs)
        return wrapped
    return decorate


def _jwt_settings(*, signing: bool = False):
    algorithm = os.getenv("API_JWT_ALGORITHM", "HS256")
    if algorithm not in {"HS256", "RS256", "RS384", "RS512"}:
        raise HTTPException(status_code=503, detail="Unsupported JWT signing algorithm")
    if algorithm.startswith("RS"):
        key = os.getenv("API_JWT_PRIVATE_KEY" if signing else "API_JWT_PUBLIC_KEY")
    else:
        key = os.getenv("API_JWT_SECRET")
    if not key or (algorithm == "HS256" and len(key) < 32):
        raise HTTPException(status_code=503, detail="JWT verification is not configured securely")
    return key, algorithm


def _password_matches(password: str, encoded: str) -> bool:
    import hashlib
    import hmac
    try:
        salt, expected = encoded.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310_000).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, AttributeError):
        return False


def _current_user(token: str = Depends(oauth2_scheme)) -> dict:
    if jwt is None:
        raise HTTPException(status_code=503, detail="Install python-jose to enable JWT authentication")
    key, algorithm = _jwt_settings()
    try:
        options = {"verify_aud": bool(os.getenv("API_JWT_AUDIENCE"))}
        claims = jwt.decode(token, key, algorithms=[algorithm],
                            issuer=os.getenv("API_JWT_ISSUER") or None,
                            audience=os.getenv("API_JWT_AUDIENCE") or None,
                            options=options)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired bearer token",
                            headers={"WWW-Authenticate": "Bearer"}) from exc
    roles = claims.get("roles", [])
    if isinstance(roles, str):
        roles = roles.split()
    if not isinstance(roles, list) or not set(roles).issubset({"trader", "risk_analyst", "admin"}):
        raise HTTPException(status_code=403, detail="Token has invalid role claims")
    return {"subject": claims.get("sub"), "roles": roles, "claims": claims}


def require_roles(*allowed_roles: str):
    def dependency(user: dict = Depends(_current_user)) -> dict:
        if "admin" not in user["roles"] and not set(user["roles"]).intersection(allowed_roles):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user
    return dependency


def _configured_users() -> dict:
    try:
        users = json.loads(os.getenv("API_USERS_JSON", "{}"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="API_USERS_JSON must be valid JSON") from exc
    return users if isinstance(users, dict) else {}


@app.post("/oauth/token")
def issue_access_token(form: OAuth2PasswordRequestForm = Depends()):
    """Issue a short-lived JWT for a configured service user (PBKDF2 password hashes)."""
    if jwt is None:
        raise HTTPException(status_code=503, detail="Install python-jose to enable JWT authentication")
    key, algorithm = _jwt_settings(signing=True)
    user = _configured_users().get(form.username)
    if not user or not _password_matches(form.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    roles = user.get("roles", [])
    if not roles or not set(roles).issubset({"trader", "risk_analyst", "admin"}):
        raise HTTPException(status_code=403, detail="Configured user has invalid roles")
    from datetime import timedelta, timezone
    now = datetime.now(timezone.utc)
    claims = {"sub": form.username, "roles": roles,
              "iat": int(now.timestamp()),
              "exp": int((now + timedelta(minutes=30)).timestamp())}
    if issuer := os.getenv("API_JWT_ISSUER"):
        claims["iss"] = issuer
    if audience := os.getenv("API_JWT_AUDIENCE"):
        claims["aud"] = audience
    return {"access_token": jwt.encode(claims, key, algorithm=algorithm), "token_type": "bearer",
            "expires_in": 1800}


async def _get_redis():
    global _redis
    if _redis is None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise HTTPException(status_code=503, detail="Install redis to enable live streams") from exc
        _redis = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True,
                                socket_connect_timeout=0.1, socket_timeout=0.1,
                                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100")))
    return _redis


@app.on_event("shutdown")
async def close_redis_client():
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
    if _interaction_producer is not None:
        await _interaction_producer.close()


@app.on_event("startup")
async def configure_tracing():
    """Export spans through OTLP when an endpoint is configured."""
    from src.observability import configure_tracing as configure_otel
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and not configure_otel("netflix-recommendation-api"):
        logger.warning("OTLP tracing requested but SDK/exporter is unavailable or already configured")


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    """Expose low-cardinality service metrics for Prometheus scraping."""
    from src.observability import render_metrics
    try:
        return Response(render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def publish_risk_event(event_type: str, payload: dict) -> float:
    """Publish risk/circuit-breaker events and return publish latency in milliseconds."""
    if event_type not in {"risk_metrics", "circuit_breaker"}:
        raise ValueError("Unsupported risk event type")
    started = time.perf_counter()
    redis = await _get_redis()
    event = {"type": event_type, "timestamp": datetime.utcnow().isoformat() + "Z", "payload": payload}
    with _span("redis.publish.risk_event"):
        await redis.publish(os.getenv("RISK_EVENTS_CHANNEL", "alpha:risk:events"), json.dumps(event))
    return (time.perf_counter() - started) * 1000


class RecommendationContext(BaseModel):
    device_type: str = Field("unknown", max_length=32)
    session_duration_seconds: float = Field(0.0, ge=0, le=86400)
    hour: int = Field(20, ge=0, le=23)
    weekend: bool = False


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=1000)


class ConversationalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    history: List[ConversationMessage] = Field(default_factory=list, max_length=12)


class StreamTelemetry(BaseModel):
    rendition: Literal["360p", "720p", "1080p", "2160p", "other"] = "other"
    bitrate_kbps: float = Field(..., ge=0, le=100000)


@app.post("/api/v1/interactions", status_code=202)
async def ingest_interaction(event: InteractionEvent,
                             user: dict = Depends(require_roles("trader"))):
    """Validate and durably publish a user interaction to the Kafka event topic."""
    global _interaction_producer
    if "admin" not in user["roles"] and user.get("subject") != event.user_id:
        raise HTTPException(status_code=403, detail="Cannot publish events for another user")
    try:
        async with _interaction_producer_lock:
            if _interaction_producer is None:
                _interaction_producer = InteractionProducer()
                await _interaction_producer.start()
        payload = await _interaction_producer.publish(event)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Kafka interaction publish failed")
        raise HTTPException(status_code=503, detail="Interaction event broker is unavailable") from exc
    return {"status": "accepted", "event_id": payload["event_id"]}


@app.post("/api/v1/telemetry/stream", status_code=202)
async def ingest_stream_telemetry(telemetry: StreamTelemetry,
                                  user: dict = Depends(require_roles("trader"))):
    """Record the active HLS bitrate reported by an authenticated player."""
    from src.observability import set_stream_bitrate
    set_stream_bitrate(telemetry.rendition, telemetry.bitrate_kbps)
    return {"status": "accepted"}


@app.post("/api/v1/recommendations/{user_id}")
async def get_personalized_recommendations(user_id: str, context: RecommendationContext,
                                           user: dict = Depends(require_roles("trader"))):
    """Return continue-watching and personalized rails after ANN retrieval/ranking."""
    if "admin" not in user["roles"] and user.get("subject") != user_id:
        raise HTTPException(status_code=403, detail="Cannot request another user's recommendations")
    try:
        return await _build_recommendation(user_id, context.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Personalized recommendation request failed")
        raise HTTPException(status_code=503, detail="Recommendation service is unavailable") from exc


async def _build_recommendation(user_id: str, context: dict) -> dict:
    from pathlib import Path
    from src.recommendations.engine import RecommendationEngine
    from src.recommendations.ranking import ContextualRanker
    from src.recommendations.two_tower import TwoTowerModel

    artifact_dir = Path(os.getenv("RECOMMENDATION_ARTIFACT_DIR", "models/recommendations"))
    model_path = artifact_dir / "two_tower.json"
    if not model_path.is_file():
        raise RuntimeError("Train and publish the recommendation model first")
    redis = await _get_redis()
    records = await redis.xrevrange(f"netflix:user:{user_id}:history", count=1000)
    events = []
    for _, fields in reversed(records):
        try:
            events.append(json.loads(fields["event"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    from src.observability import record_cache_result
    record_cache_result(bool(records))
    tower = TwoTowerModel.load(model_path)
    ranker_path = artifact_dir / "contextual_ranker.joblib"
    ranker = ContextualRanker.load(ranker_path) if ranker_path.is_file() else ContextualRanker()
    engine = RecommendationEngine(tower, ranker)
    return await asyncio.to_thread(engine.recommend, user_id, events=events, context=context)


@app.get("/api/v1/recommendations/stream")
async def stream_recommendations(request: Request, device_type: str = "unknown",
                                 session_duration_seconds: float = 0.0, hour: int = 20,
                                 weekend: bool = False,
                                 user: dict = Depends(require_roles("trader"))):
    """Send an initial recommendation and refreshed rows over authenticated SSE."""
    if EventSourceResponse is None:
        raise HTTPException(status_code=503, detail="Install sse-starlette to enable SSE")
    user_id = user.get("subject")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authenticated user subject is required")
    context = RecommendationContext(device_type=device_type,
        session_duration_seconds=session_duration_seconds, hour=hour, weekend=weekend).model_dump()
    redis = await _get_redis()
    channel = f"netflix:recommendations:refresh:{user_id}"
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    try:
        initial = await _build_recommendation(user_id, context)
    except RuntimeError as exc:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        logger.exception("Initial recommendation stream snapshot failed")
        raise HTTPException(status_code=503, detail="Recommendation service is unavailable") from exc

    async def events():
        try:
            yield {"event": "recommendation", "data": json.dumps(initial, default=str)}
            while not await request.is_disconnected():
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    refreshed = await _build_recommendation(user_id, context)
                    yield {"event": "recommendation", "data": json.dumps(refreshed, default=str)}
                except Exception:
                    logger.exception("Live recommendation refresh failed for user %s", user_id)
                    from src.recommendations.explain import response as explain_response
                    error = explain_response(verdict="Recommendation refresh unavailable",
                        cosine_similarity=None,
                        rationale="A new personalized row could not be computed after the latest interaction.",
                        recommendation="Keep the current rows and retry after the recommendation service recovers.")
                    yield {"event": "error", "data": json.dumps(error)}
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return EventSourceResponse(events(), ping=15, send_timeout=5)


@app.post("/api/v1/search/recommendations")
async def search_catalog_recommendations(request: ConversationalSearchRequest,
                                         user: dict = Depends(require_roles("trader"))):
    """Search the catalog conversationally with LangGraph, Qdrant, and Guardrails."""
    from src.copilot.search_agent import get_search_agent
    from src.recommendations.explain import response as explain_response
    try:
        agent = await asyncio.to_thread(get_search_agent)
        history = [message.model_dump() for message in request.history]
        result = await asyncio.to_thread(agent.search, request.query, history)
        matches = result.get("recommendations", [])
        top = matches[0] if matches else None
        if top:
            score = top.get("score")
            title = str(top.get("title") or "Top catalog match")
            genres = top.get("genres") or []
            genre_text = ", ".join(map(str, genres)) or "unspecified genres"
            rationale = (f"Matched the request {request.query!r}; the catalog vector search selected "
                         f"{title!r} from {genre_text}.")
            recommendation = "Show the catalog matches in ranked order: " + ", ".join(
                str(item.get("title") or "Untitled") for item in matches[:8]) + "."
            verdict = title
        else:
            score = None
            rationale = f"No catalog vector matches were found for the request {request.query!r}."
            recommendation = "Ask for a different genre, runtime, or mood, then search the catalog again."
            verdict = "No matching title"
        return explain_response(verdict=verdict,
                                cosine_similarity=float(score) if score is not None else None,
                                rationale=rationale, recommendation=recommendation)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Conversational catalog search failed")
        raise HTTPException(status_code=503, detail="Catalog search is unavailable") from exc


@app.post("/api/v1/catalog/upload")
async def upload_catalog_asset(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                               user: dict = Depends(require_roles("admin"))):
    """Accept a catalog asset and enqueue its local processing as a background task.

    Videos are HLS-transcoded with FFmpeg; text sidecars are parsed and indexed
    when QDRANT_URL is configured. Uploads are bounded by CATALOG_MAX_UPLOAD_BYTES.
    """
    from pathlib import Path
    from src.catalog.asset_pipeline import MAX_UPLOAD_BYTES, new_asset_destination

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in {".mp4", ".mkv", ".mov", ".json", ".srt", ".vtt"}:
        raise HTTPException(status_code=415, detail="Supported uploads: MP4, MKV, MOV, JSON, SRT, VTT")
    try:
        asset_id, destination = new_asset_destination(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                output.write(chunk)
        if not written:
            raise HTTPException(status_code=400, detail="Empty uploads are not accepted")
        if extension in {".json", ".srt", ".vtt"} and written > int(os.getenv("CATALOG_MAX_TEXT_BYTES", str(20 * 1024**2))):
            raise HTTPException(status_code=413, detail="Text metadata uploads are limited to 20 MiB")
        (destination.parent / "status.json").write_text(
            json.dumps({"asset_id": asset_id, "status": "processing", "bytes": written}),
            encoding="utf-8",
        )
        background_tasks.add_task(_process_catalog_asset_job, filename, destination, asset_id, written)
        return {"asset_id": asset_id, "status": "processing", "bytes": written}
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except (ValueError, RuntimeError) as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Catalog upload processing failed for asset %s", asset_id)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Catalog asset processing failed") from exc
    finally:
        await file.close()


def _process_catalog_asset_job(filename: str, destination, asset_id: str, size_bytes: int) -> None:
    """Persist the outcome so clients can poll after the upload request returns."""
    from src.catalog.asset_pipeline import process_saved_upload

    status_path = destination.parent / "status.json"
    try:
        result = process_saved_upload(filename, destination)
        result.update({"status": "ready", "bytes": size_bytes})
        status_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.exception("Catalog asset processing failed for asset %s", asset_id)
        status_path.write_text(json.dumps({"asset_id": asset_id, "status": "failed",
                                          "detail": str(exc)[:1000]}, ensure_ascii=False), encoding="utf-8")


@app.get("/api/v1/catalog/assets/{asset_id}")
def catalog_asset_status(asset_id: str, user: dict = Depends(require_roles("admin"))):
    """Read persisted processing status for a previously uploaded asset."""
    import uuid
    from src.catalog.asset_pipeline import UPLOAD_ROOT

    try:
        safe_id = str(uuid.UUID(asset_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Asset not found") from exc
    status_file = UPLOAD_ROOT / safe_id / "status.json"
    if not status_file.is_file():
        raise HTTPException(status_code=404, detail="Asset processing status not found")
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read asset processing status") from exc


@app.get("/api/v1/catalog/assets/{asset_id}/hls/{hls_path:path}")
def catalog_hls_asset(asset_id: str, hls_path: str,
                      user: dict = Depends(require_roles("trader"))):
    """Serve a validated asset HLS playlist or segment to authenticated players."""
    import uuid
    from pathlib import Path
    from src.catalog.asset_pipeline import UPLOAD_ROOT

    try:
        safe_id = str(uuid.UUID(asset_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="HLS asset not found") from exc
    root = (UPLOAD_ROOT / safe_id / "hls").resolve()
    requested = (root / Path(hls_path)).resolve()
    if not requested.is_relative_to(root) or requested.suffix.lower() not in {".m3u8", ".ts", ".m4s", ".mp4", ".aac"}:
        raise HTTPException(status_code=404, detail="HLS asset not found")
    if not requested.is_file():
        raise HTTPException(status_code=404, detail="HLS asset not found")
    media_type = "application/vnd.apple.mpegurl" if requested.suffix.lower() == ".m3u8" else "video/mp2t"
    return FileResponse(requested, media_type=media_type, headers={"Cache-Control": "private, max-age=60"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded

    _rate = os.getenv("API_RATE_LIMIT", "10")
    limiter = Limiter(key_func=get_remote_address,
                      default_limits=[f"{_rate}/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    _rate_limiting = True
except ImportError:
    _rate_limiting = False
    logger.warning("slowapi not installed — rate limiting disabled")

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
try:
    _model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    _model = None
    logger.error(f"Model load failed: {e}")

# ── Schemas ───────────────────────────────────────────────────────────────────
class OHLCVRow(BaseModel):
    open:   float = Field(..., example=645.0)
    high:   float = Field(..., example=655.0)
    low:    float = Field(..., example=640.0)
    close:  float = Field(..., example=650.0)
    volume: float = Field(..., example=5_000_000)


class PredictRequest(BaseModel):
    rows:   List[OHLCVRow] = Field(..., min_items=10,
                description="Last N trading days of OHLCV, most recent last")
    ticker: Optional[str]  = Field("NFLX", description="Ticker symbol")


class PredictResponse(BaseModel):
    ticker:               str
    predicted_return_pct: float
    predicted_next_close: float
    last_close:           float
    signal:               str
    confidence_interval:  Optional[dict] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":         "ok",
        "model_loaded":   _model is not None,
        "rate_limiting":  _rate_limiting,
        "version":        "2.0.0",
    }


@app.get("/tickers")
def tickers():
    """List of supported tickers (any valid yfinance symbol)."""
    return {
        "note":    "Any valid Yahoo Finance ticker is supported via live data",
        "popular": ["NFLX", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META"],
    }


@app.post("/predict", response_model=PredictResponse)
@traced("model.inference")
def predict(request: PredictRequest, user: dict = Depends(require_roles("trader", "risk_analyst"))):
    if _model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run python main.py first.")
    try:
        df = pd.DataFrame([r.dict() for r in request.rows])
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = compute_features_from_ohlcv(df)

        train_feats = getattr(_model, "feature_names_", FEATURES)
        for f in train_feats:
            if f not in df.columns:
                df[f] = 0.0
        row        = df[train_feats].iloc[[-1]]
        pred_ret   = float(_model.predict(row)[0])
        last_close = request.rows[-1].close
        pred_price = last_close * (1 + pred_ret / 100)
        signal     = "BUY" if pred_ret > 0 else "HOLD"

        ci = None
        if hasattr(_model, "conformal_"):
            cp = _model.conformal_
            lo_r, hi_r = cp.predict_interval(row)
            nominal_coverage = 1.0 - float(getattr(cp, "alpha", 0.1))
            ci = {
                "lower_return_pct": round(float(lo_r[0]), 4),
                "upper_return_pct": round(float(hi_r[0]), 4),
                "lower_price":      round(last_close * (1 + lo_r[0] / 100), 2),
                "upper_price":      round(last_close * (1 + hi_r[0] / 100), 2),
                "coverage":         f"{nominal_coverage:.0%}",
                "nominal_coverage": nominal_coverage,
            }

        return PredictResponse(
            ticker               = request.ticker or "NFLX",
            predicted_return_pct = round(pred_ret, 4),
            predicted_next_close = round(pred_price, 2),
            last_close           = last_close,
            signal               = signal,
            confidence_interval  = ci,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stream/predictions/{symbol}")
async def stream_predictions(symbol: str, user: dict = Depends(require_roles("trader", "risk_analyst"))):
    """Stream Redis price events and predictions with calibrated conformal intervals.

    The market-data publisher should send JSON on ``alpha:stream:{SYMBOL}``.
    Events with a ``rows`` field (last ten or more OHLCV rows) trigger inference;
    every event is also forwarded as a price/update event.
    """
    if EventSourceResponse is None:
        raise HTTPException(status_code=503, detail="Install sse-starlette to enable SSE")
    ticker = symbol.upper()
    redis = await _get_redis()
    from src.recommendations.explain import response as explain_response

    def stream_envelope(*, verdict: str, rationale: str, recommendation: str) -> dict:
        return explain_response(verdict=verdict, cosine_similarity=None,
                                rationale=rationale, recommendation=recommendation)

    async def events():
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"alpha:stream:{ticker}")
        try:
            connected = stream_envelope(verdict=f"{ticker} stream connected",
                rationale="The stream is connected and waiting for the next market update.",
                recommendation=f"Monitor incoming {ticker} market updates and prediction events.")
            yield {"event": "connected", "data": json.dumps(connected)}
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    update = json.loads(message["data"])
                except (TypeError, json.JSONDecodeError):
                    logger.warning("Discarding malformed market update for %s", ticker)
                    continue
                price_update = {key: value for key, value in update.items() if key != "rows"}
                price = stream_envelope(verdict=f"{ticker} market update",
                    rationale=f"Received market update: {json.dumps(price_update, default=str)}",
                    recommendation=f"Use the latest {ticker} market update as context for the next prediction.")
                yield {"event": "price", "data": json.dumps(price, default=str)}
                rows = update.get("rows")
                if not rows or _model is None:
                    continue
                try:
                    request = PredictRequest(rows=[OHLCVRow(**row) for row in rows], ticker=ticker)
                    prediction = await asyncio.to_thread(predict, request)
                    result = prediction.dict()
                    ci = result.get("confidence_interval") or {}
                    interval_text = (f" The calibrated return interval is {ci['lower_return_pct']:.4f}% to "
                                     f"{ci['upper_return_pct']:.4f}%."
                                     if ci.get("lower_return_pct") is not None and ci.get("upper_return_pct") is not None
                                     else " No calibrated confidence interval is available.")
                    result = stream_envelope(
                        verdict=f"{ticker}: {result['signal']} outlook",
                        rationale=(f"The streaming model predicts a {result['predicted_return_pct']:+.4f}% return "
                                   f"from a last close of {result['last_close']:.2f}.{interval_text}"),
                        recommendation=(f"Review the {ticker} prediction and its confidence interval before "
                                        "using it in a trading decision."),
                    )
                    yield {"event": "prediction", "data": json.dumps(result, default=str)}
                except Exception:
                    logger.exception("Streaming inference failed for %s", ticker)
                    error = stream_envelope(verdict=f"{ticker} prediction unavailable",
                        rationale="Streaming inference failed for this market update.",
                        recommendation="Wait for the next valid market update before requesting a prediction.")
                    yield {"event": "error", "data": json.dumps(error)}
        finally:
            await pubsub.unsubscribe(f"alpha:stream:{ticker}")
            await pubsub.aclose()

    return EventSourceResponse(events(), ping=15, send_timeout=5)


@app.get("/api/v1/stream/risk")
async def stream_risk_events(user: dict = Depends(require_roles("risk_analyst"))):
    """Stream risk metrics and circuit-breaker broadcasts from Redis Pub/Sub."""
    if EventSourceResponse is None:
        raise HTTPException(status_code=503, detail="Install sse-starlette to enable SSE")
    redis = await _get_redis()

    async def events():
        pubsub = redis.pubsub()
        channel = os.getenv("RISK_EVENTS_CHANNEL", "alpha:risk:events")
        await pubsub.subscribe(channel)
        try:
            yield {"event": "connected", "data": json.dumps({"channel": channel})}
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                yield {"event": "risk", "data": str(message["data"])}
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return EventSourceResponse(events(), ping=15, send_timeout=5)


@app.get("/features")
def features():
    return {"features": FEATURES, "count": len(FEATURES)}


@app.get("/model_info")
def model_info():
    info: dict = {
        "model_type":   "ManualStackingRegressor",
        "architecture": "XGBoost + LightGBM + RandomForest + ExtraTrees -> Ridge",
        "target":       "next-day return (%)",
        "n_features":   len(FEATURES),
        "model_loaded": _model is not None,
        "version":      "2.0.0",
    }
    if _model is not None and hasattr(_model, "feature_names_"):
        info["trained_feature_count"] = len(_model.feature_names_)
    if _model is not None and hasattr(_model, "conformal_"):
        cp = _model.conformal_
        info["conformal_alpha"] = cp.alpha
        info["conformal_width"] = cp.interval_width
    metrics_path = os.path.join(ROOT, "outputs", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            info["latest_metrics"] = json.load(f)
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        info["model_trained_at"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    # Registry info
    try:
        from src.model_registry import get_registry
        reg = get_registry()
        info["total_versions"] = len(reg["models"])
        info["latest_version"] = reg.get("latest")
    except Exception:
        pass

    return info


@app.get("/registry")
def registry():
    """Return full model version registry."""
    try:
        from src.model_registry import get_registry
        return get_registry()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Risk & Execution endpoints ────────────────────────────────────────────────

class RiskRequest(BaseModel):
    ticker:           str   = Field("NFLX")
    pred_return_pct:  float = Field(..., description="Model predicted return (%)")
    last_price:       float = Field(..., description="Current price")
    atr:              float = Field(..., description="14-period ATR")
    portfolio_value:  float = Field(100_000.0, description="Total portfolio ($)")
    win_rate:         float = Field(0.52, description="Historical win rate (0-1)")
    avg_win_pct:      float = Field(1.5)
    avg_loss_pct:     float = Field(1.0)
    max_position_pct: float = Field(0.05, description="Max position size (0-1)")
    max_drawdown_halt:float = Field(0.10, description="Circuit breaker drawdown")
    current_portfolio_value: Optional[float] = Field(None, gt=0,
        description="Latest marked portfolio value used to evaluate the drawdown breaker")


class ExecuteRequest(BaseModel):
    ticker:      str   = Field("NFLX")
    shares:      int   = Field(..., gt=0)
    side:        str   = Field("buy", description="buy or sell")
    order_type:  str   = Field("market", description="market or limit")
    limit_price: Optional[float] = Field(None)
    broker:      str   = Field("alpaca", description="alpaca | paper")


@app.post("/risk/position")
async def compute_risk_position(request: RiskRequest,
                               user: dict = Depends(require_roles("trader", "risk_analyst"))):
    """
    Compute execution-ready position size with full risk controls.
    Returns stop-loss, take-profit, shares, risk per trade.
    """
    try:
        from src.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(
            portfolio_value    = request.portfolio_value,
            max_position_pct   = request.max_position_pct,
            max_drawdown_halt  = request.max_drawdown_halt,
        )
        rm    = RiskManager(cfg)
        if request.current_portfolio_value is not None:
            rm.update_portfolio_value(request.current_portfolio_value)
        order = rm.compute_position(
            ticker      = request.ticker,
            pred_return = request.pred_return_pct,
            last_price  = request.last_price,
            atr         = request.atr,
            win_rate    = request.win_rate,
            avg_win_pct = request.avg_win_pct,
            avg_loss_pct= request.avg_loss_pct,
        )
        result = order.to_dict()
        event_type = "circuit_breaker" if order.signal == "HALT" else "risk_metrics"
        try:
            latency_ms = await publish_risk_event(event_type, result)
            result["redis_publish_latency_ms"] = round(latency_ms, 3)
            logger.info("Published %s event to Redis in %.3f ms", event_type, latency_ms)
        except Exception:
            logger.exception("Risk result generated but Redis broadcast failed")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk/matrix")
def risk_matrix(request: RiskRequest, user: dict = Depends(require_roles("trader", "risk_analyst"))):
    """Return full risk matrix for a given prediction."""
    try:
        from src.risk_manager import RiskManager, RiskConfig
        cfg = RiskConfig(portfolio_value=request.portfolio_value,
                         max_position_pct=request.max_position_pct)
        rm  = RiskManager(cfg)
        return rm.risk_matrix(
            pred_return    = request.pred_return_pct,
            last_price     = request.last_price,
            atr            = request.atr,
            portfolio_value= request.portfolio_value,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute")
@traced("broker.execute_order")
def execute_trade(request: ExecuteRequest,
                  user: dict = Depends(require_roles("trader"))):
    """
    Execute a trade via broker integration.
    Supports: Alpaca (paper + live), paper simulation.

    Required env vars for Alpaca:
      ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    """
    # HITL gate
    try:
        from src.copilot.hitl_router import HITLRouter, make_signal_id
        _pos_val = float(getattr(request, 'position_value', 0) or 0)
        if _pos_val > 0:
            _hitl = HITLRouter()
            _pending = [p for p in _hitl.pending_approvals() if p.get('ticker') == getattr(request, 'ticker', '')]
            if _pending:
                raise HTTPException(status_code=403, detail={"error": "HITL_REQUIRED", "signal_id": _pending[0]['signal_id']})
    except HTTPException:
        raise
    except Exception:
        pass

    broker = request.broker.lower()

    if broker == "paper":
        # Simulated paper execution — no real broker needed
        return {
            "status":     "filled",
            "broker":     "paper",
            "ticker":     request.ticker,
            "side":       request.side,
            "shares":     request.shares,
            "order_type": request.order_type,
            "fill_price": request.limit_price,
            "message":    "Paper trade executed (simulation only)",
        }

    if broker == "alpaca":
        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise HTTPException(
                status_code=503,
                detail="Alpaca keys not configured. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        try:
            import requests as req
            headers = {
                "APCA-API-KEY-ID":     api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Content-Type":        "application/json",
            }
            body: dict = {
                "symbol":        request.ticker,
                "qty":           str(request.shares),
                "side":          request.side,
                "type":          request.order_type,
                "time_in_force": "day",
            }
            if request.order_type == "limit" and request.limit_price:
                body["limit_price"] = str(request.limit_price)

            resp = req.post(f"{base_url}/v2/orders",
                            json=body, headers=headers, timeout=10)

            if resp.status_code in (200, 201):
                data = resp.json()
                return {
                    "status":     data.get("status", "submitted"),
                    "broker":     "alpaca",
                    "order_id":   data.get("id"),
                    "ticker":     request.ticker,
                    "side":       request.side,
                    "shares":     request.shares,
                    "order_type": request.order_type,
                    "fill_price": data.get("filled_avg_price"),
                    "message":    "Order submitted to Alpaca",
                }
            else:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Alpaca error: {resp.text}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execution failed: {e}")

    raise HTTPException(status_code=400, detail=f"Unknown broker: {broker}. Use 'alpaca' or 'paper'")


# ── Alpha Engine Pro v2.0 endpoints ──────────────────────────────────────────
import traceback as _tb


@app.get("/strategies")
def list_strategies():
    try:
        from src.strategy_registry import StrategyRegistry
        reg = StrategyRegistry()
        names = reg.list_strategies()
        strategies = []
        for n in names:
            cfg = reg.get(n)
            strategies.append({
                "name": cfg.name,
                "tickers": cfg.tickers,
                "feature_set_count": len(cfg.feature_set),
                "tenant_id": cfg.tenant_id,
                "description": cfg.description,
            })
        return {"strategies": strategies}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/strategies/{name}/run")
@traced("strategy.run_inference")
def run_strategy(name: str, source: str = "csv",
                 user: dict = Depends(require_roles("trader"))):
    try:
        from src.strategy_registry import StrategyRegistry
        return StrategyRegistry().run_strategy(name, source=source)
    except KeyError:
        raise HTTPException(404, f"Strategy '{name}' not found")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/track-record")
def get_track_record(strategy: str = "nflx_momentum", ticker: str = "", tenant_id: str = "default"):
    try:
        from src.track_record import TrackRecord
        tr = TrackRecord(strategy_name=strategy)
        df = tr.load()
        if ticker and "ticker" in df.columns:
            df = df[df["ticker"] == ticker]
        metrics = tr.compute_metrics()
        return {"strategy": strategy, "metrics": metrics, "n_rows": len(df)}
    except Exception as e:
        raise HTTPException(500, str(e))


class CopilotRequest(BaseModel):
    ticker: str = "NFLX"
    strategy_name: str = "nflx_momentum"
    query: str = ""
    position_value: float = 0.0


@app.post("/copilot/research")
@traced("copilot.research_inference")
def copilot_research(req: CopilotRequest,
                     user: dict = Depends(require_roles("trader", "risk_analyst"))):
    try:
        from src.copilot import CopilotGraph
        g = CopilotGraph(ticker=req.ticker)
        result = g.run(ticker=req.ticker, query=req.query or None, position_value=req.position_value)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/hitl/pending")
def hitl_pending(user: dict = Depends(require_roles("risk_analyst"))):
    try:
        from src.copilot.hitl_router import HITLRouter
        return {"pending": HITLRouter().pending_approvals()}
    except Exception as e:
        raise HTTPException(500, str(e))


class HITLApprovalRequest(BaseModel):
    approver: str = "human"


@app.post("/hitl/approve/{signal_id}")
def hitl_approve(signal_id: str, req: HITLApprovalRequest,
                 user: dict = Depends(require_roles("admin"))):
    try:
        from src.copilot.hitl_router import HITLRouter
        return HITLRouter().approve(signal_id, approver=req.approver)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/compliance/report")
def compliance_report(tenant_id: str = "default",
                     user: dict = Depends(require_roles("risk_analyst"))):
    try:
        from src.compliance_sebi import SEBIComplianceChecker
        checker = SEBIComplianceChecker()
        return checker.generate_report()
    except Exception as e:
        raise HTTPException(500, str(e))

