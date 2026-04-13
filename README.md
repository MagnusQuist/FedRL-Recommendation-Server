# FedRL — Recommendation Server

Federated RL recommendation server for the **Nudge2Green** thesis project.  
Exposes a food catalogue REST API and learning aggregation endpoints for Raspberry Pi clients running LinTS (Linear Thompson Sampling) in one of three experiment modes.

---

## Experiment modes

| Mode | Description | Server involvement |
|------|-------------|-------------------|
| **federated** | Clients retrain locally and upload backbone weights; server aggregates via FedAvg | `GET/POST /api/v1/backbone/model` |
| **centralized** | Clients upload raw interaction tuples; server trains the model centrally | `GET /api/v1/centralized/model`, `POST /api/v1/centralized/interactions` |
| **control** | No model, no server communication | None |

Both `federated` and `centralized` are initialised from the **same pretrained backbone seed** (`BACKBONE_INIT_SEED`, default `42`) so the experiment starts from an identical baseline. A single server instance handles all client modes simultaneously.

---

## Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| ORM | SQLAlchemy 2 (async) |
| ML | PyTorch (centralized retraining), NumPy (FedAvg) |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions → ghcr.io |

---

## Quick start (local development)

### 1. Clone and configure

```bash
git clone https://github.com/MagnusQuist/FedRL-Recommendation-Server.git
cd FedRL-Recommendation-Server
cp .env.example .env
```

### 2. Start all services

```bash
docker compose up --build
```

This starts:
- **PostgreSQL** on `localhost:5432`
- **FastAPI server** on `localhost:8000`
- **pgAdmin** on `localhost:5050` (user: `admin@fedrl.local` / `admin`)

### 3. Database schema

On startup the server runs `Base.metadata.create_all()` when `AUTO_CREATE_MODELS=true` (default in Docker Compose). This **creates missing tables** but does not alter existing ones. For schema changes on an existing database, apply manual SQL or wipe the volume.

### 4. Seed the food catalogue

The catalogue is seeded from three JSON files under `data/` by [`app/db/seed_catalogue.py`](app/db/seed_catalogue.py).

**Startup behaviour:** `AUTO_SEED_DATA_ON_STARTUP=true` (default) seeds backbone + catalogue in the background after the server starts. Set to `false` if seeding manually.

```bash
# Seed manually (inside the server container or with DATABASE_URL set):
docker compose exec server python -m app.db.seed_catalogue
```

### 5. Verify

```bash
curl http://localhost:8000/api/v1/health
# → {"status": "ok", "database": "reachable"}
```

Interactive API docs: **http://localhost:8000/docs**

---

## API endpoints

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Server + database status |
| `GET` | `/api/v1/seed-status` | Whether the database has been seeded |

### Catalogue

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/catalogue/snapshot` | Full catalogue snapshot (items, categories, substitution groups, version) |
| `GET` | `/api/v1/catalogue/version` | Current catalogue version |

### Images

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/images/food_item/{id}` | JPEG for a food item |
| `GET` | `/api/v1/images/product_label/{name}` | WebP for a product label |

### Federated backbone

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/backbone/model?since=<v>&algorithm=ts` | Download global backbone if newer than `since`; returns 304 otherwise |
| `POST` | `/api/v1/backbone/model` | Upload local backbone weights for FedAvg aggregation |
| `GET` | `/api/v1/backbone/version?algorithm=ts` | Current global backbone version |
| `GET` | `/api/v1/backbone/status?algorithm=ts` | Aggregation queue status (queued clients, rounds completed, timeout) |

**POST body:**
```json
{
  "client_id": "client_01",
  "algorithm": "ts",
  "backbone_version": 3,
  "interaction_count": 10,
  "backbone_weights": "<gzip+b64>"
}
```

**GET response (200):**
```json
{ "version": 3, "algorithm": "ts", "client_count": 2, "total_interactions": 40, "backbone_weights": "<gzip+b64>" }
```

### Centralized training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/centralized/interactions` | Upload raw interaction tuples for centralized training |
| `GET` | `/api/v1/centralized/model?since=<v>&algorithm=ts` | Download centralized backbone + head weights if newer than `since`; returns 304 otherwise |

**POST body:**
```json
{
  "client_id": "client_01",
  "algorithm": "ts",
  "count": 10,
  "data": "<gzip+b64 encoded JSON array of interaction tuples>"
}
```

**POST response (200):**
```json
{ "accepted": true, "server_model_version": 8 }
```

**GET response (200):**
```json
{
  "model_version": 8,
  "backbone_weights": "<gzip+b64>",
  "head_weights": {
    "item":  "<gzip+b64>",
    "price": "<gzip+b64>",
    "nudge": "<gzip+b64>"
  }
}
```

### Developer

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/dev/metrics/json` | FL aggregation state snapshot |
| `WS` | `/api/v1/dev/metrics/ws` | Live metrics stream (pushes every 2 s) |
| `GET` | `/api/v1/dev/metrics` | HTML dashboard |

---

## Federated learning

FedAvg is triggered per algorithm when either condition is met:

- **Client threshold**: `FL_MIN_CLIENTS_PER_ROUND` uploads queued (default `2`)
- **Timeout**: `FL_ROUND_TIMEOUT_SECONDS` elapsed since first upload (default `60`)

Aggregated backbone versions are persisted to PostgreSQL (`global_backbone_versions` table). If fewer than `FL_MIN_CLIENTS_PER_ROUND` clients have uploaded when the timeout fires, uploads are carried forward to the next round.

---

## Centralized training

On each `POST /centralized/interactions`:

1. Decodes the gzip+b64 tuple batch.
2. Appends tuples to a persistent pool (capped at `CENTRALIZED_MAX_TUPLES`, default `2000`).
3. Retrains the backbone on the full pool (3 epochs, Adam lr=1e-3).
4. Applies a Bayesian online update to the global item, price, and nudge heads for every tuple.
5. Increments `model_version` and persists all state to disk under `data/centralized/`.

State is fully restored from disk on server restart — no training progress is lost.

---

## Project structure

```
.
├── app/
│   ├── main.py                    # ASGI entry point
│   ├── api/
│   │   ├── app.py                 # App factory + lifespan (aggregator + centralized service init)
│   │   ├── routers/
│   │   │   ├── api.py             # Root /api/v1 router
│   │   │   ├── backbone.py        # Federated backbone endpoints
│   │   │   ├── centralized.py     # Centralized training endpoints
│   │   │   ├── catalogue.py
│   │   │   ├── health.py
│   │   │   ├── images.py
│   │   │   └── metrics.py         # Dev dashboard
│   │   └── schemas/
│   │       ├── backbone.py        # Federated request/response models
│   │       └── centralized.py     # Centralized request/response models
│   ├── db/
│   │   ├── database.py            # Async SQLAlchemy engine + session
│   │   ├── models/
│   │   │   └── backbone.py        # GlobalBackboneVersion ORM model
│   │   ├── seed_backbone.py       # Pretrained backbone seed (version 1)
│   │   └── helpers/
│   │       └── seed_db.py         # Startup seed orchestration
│   ├── fl/
│   │   ├── aggregator.py          # FedAvg aggregation service
│   │   └── centralized.py        # Centralized backbone, heads, retraining, persistence
│   ├── web/                       # Dev dashboard static assets
│   └── requirements.txt
├── data/
│   ├── categories.json
│   ├── substitution_groups.json
│   ├── product_items.json
│   └── centralized/               # Centralized model persistence (auto-created at runtime)
│       ├── backbone_centralized.pt
│       ├── reward_predictor_centralized.pt
│       ├── heads_centralized.json
│       ├── tuple_pool.json
│       └── meta.json
├── pretrain/                      # Offline backbone pretraining utilities
├── .github/workflows/             # CI/CD — build & push to ghcr.io
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `fedrl` | Database user |
| `POSTGRES_PASSWORD` | `fedrl` | Database password |
| `POSTGRES_HOST` | `localhost` | Database host |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_DB` | `fedrl` | Database name |
| `AUTO_CREATE_MODELS` | `false` | Run `create_all` on startup |
| `AUTO_SEED_DATA_ON_STARTUP` | `true` | Seed catalogue + backbone in background |
| `BACKBONE_INIT_SEED` | `42` | Random seed for pretrained backbone initialisation (shared by both federated and centralized) |
| `FL_MIN_CLIENTS_PER_ROUND` | `2` | Minimum uploads to trigger FedAvg |
| `FL_ROUND_TIMEOUT_SECONDS` | `60` | FedAvg timeout (seconds) |

---

## CI/CD

Every push to `main` builds and pushes a Docker image to the GitHub Container Registry:

```
ghcr.io/magnusquist/fedrl-recommendation-server:latest
ghcr.io/magnusquist/fedrl-recommendation-server:sha-<commit>
```

Pull requests trigger a build-only run (no push) to validate the Dockerfile.

---

## Development notes

- **No authentication** — the server assumes a trusted local network between Pi clients and the developer's machine. By design for the thesis MVP.
- **One server, three modes** — a single server process handles federated and centralized clients simultaneously. Control-mode clients do not contact the server at all.
- **Experimental fairness** — both federated and centralized backbones are seeded from identical initial weights (`BACKBONE_INIT_SEED=42`) so comparisons are valid from step 0.
- **Centralized retrains on every upload** — unlike federated which batches clients into rounds. If upload frequency differs significantly between arms this asymmetry should be considered when interpreting results.
