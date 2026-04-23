# FedRL — Recommendation Server

Federated RL recommendation server for the **Nudge2Green** thesis project.
Exposes a food catalogue REST API and training endpoints for clients running LinTS (Linear Thompson Sampling) in one of three experiment modes.

### AWS Deployment
[Guide to AWS Deployment](./aws-deployment.md)

---

## Experiment modes

| Mode | Description | Server involvement |
|------|-------------|-------------------|
| **federated** | Clients retrain locally and upload backbone weights; the server aggregates via FedAvg | `GET/POST /api/v1/federated/model` |
| **centralized** | Clients upload raw interaction tuples; the server trains the model centrally | `GET /api/v1/centralized/model`, `POST /api/v1/centralized/interactions` |
| **control** | No model, no server communication | None |

`federated` and `centralized` are initialised from the **same pretrained backbone weights** (produced by the `pretrain/` pipeline) so the arms start from an identical baseline. A single server process handles all client modes simultaneously.

---

## Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI + Uvicorn |
| Database | PostgreSQL 17 |
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
docker compose -f docker-compose.dev.yml up --build
```

This starts:
- **PostgreSQL** on `localhost:5432`
- **FastAPI server** on `localhost:8000`
- A one-shot **`seed`** container that runs `python -m app.db.seeding` before the server starts and exits

### 3. Database schema & seeding

Bootstrap is handled by the `seed` container (see [`app/db/seeding`](app/db/seeding/)). When the target database has no tables it:

1. Creates the schema via `Base.metadata.create_all()`.
2. Seeds the federated + centralized backbones from `PRETRAINED_WEIGHTS_PATH` (defaults to `app/db/seeding/data/pretrained/pretrained_backbone_weights.npz`).
3. Seeds the food catalogue from the JSON/CSV files under `app/db/seeding/data/`.

If the schema already exists the seed container is a no-op, so it's safe on every `docker compose up`. The project does not use Alembic — **schema changes on an existing database require a manual `ALTER TABLE` or wiping the `postgres_data` volume**.

### 4. Re-seeding manually

```bash
# Force re-create tables + re-run every seeder (full replace for the catalogue):
docker compose -f docker-compose.dev.yml run --rm seed python -m app.db.seeding --force

# Re-seed only parts of the database:
docker compose -f docker-compose.dev.yml run --rm seed python -m app.db.seeding --catalogue-only
docker compose -f docker-compose.dev.yml run --rm seed python -m app.db.seeding --backbones-only
```

### 5. Verify

```bash
curl http://localhost:8000/api/v1/health
# → {"status": "ok", "database": "reachable"}
```

Interactive API docs: **http://localhost:8000/docs**

---

## Production (AWS RDS)

Use `docker-compose.prod.yml`, which pulls the prebuilt image from GHCR and points at an external Postgres via `DATABASE_URL`:

```bash
docker compose -f docker-compose.prod.yml up -d
```

Required env vars in prod:
- `DATABASE_URL` — async SQLAlchemy URL for RDS (`postgresql+asyncpg://user:pw@host:5432/db`).
- `SERVER_IMAGE` (optional) — override the default GHCR image tag.

---

## API endpoints

All endpoints are mounted under `/api/v1`.

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server + database status |
| `GET` | `/seed-status` | Whether the database contains seed data |

### Catalogue

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/catalogue/snapshot` | Full catalogue snapshot (items, categories, substitution groups, version) |
| `GET` | `/catalogue/version` | Current catalogue version |

### Images

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/images/food_item/{id}` | JPEG for a food item |
| `GET` | `/images/product_label/{name}` | WebP for a product label |

### Federated backbone

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/federated/model?since=<v>` | Download the current backbone; returns 304 when `since` already matches the latest version |
| `POST` | `/federated/model` | Upload local backbone weights for FedAvg aggregation |
| `GET` | `/federated/version` | Current federated backbone version |
| `GET` | `/federated/status` | Aggregation queue state (current version, queued client ids, rounds completed, clients per round) |

**POST body:**
```json
{
  "client_id": "client_01",
  "backbone_version": 3,
  "interaction_count": 10,
  "backbone_weights": "<gzip+b64>"
}
```

`backbone_weights` must be a gzip-compressed, base64-encoded JSON dict containing **exactly** these four keys (anything else is rejected): `backbone.0.weight`, `backbone.0.bias`, `backbone.2.weight`, `backbone.2.bias`. This enforces the privacy constraint that local head weights are never transmitted.

**GET response (200):**
```json
{
  "version": 3,
  "client_count": 2,
  "total_interactions": 40,
  "backbone_weights": "<gzip+b64>"
}
```

### Centralized training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/centralized/interactions` | Upload raw interaction tuples |
| `GET`  | `/centralized/model?since=<v>` | Download the centralized backbone + head weights; returns 304 when `since` already matches the latest version |
| `GET`  | `/centralized/version` | Current centralized model version |
| `GET`  | `/centralized/status` | Training queue state (current version, queued clients, rounds completed, pool size, clients per round) |

**POST body:**
```json
{
  "client_id": "client_01",
  "count": 10,
  "data": "<gzip+b64 encoded JSON array of interaction tuples>"
}
```

**POST response (200):**
```json
{
  "accepted": true,
  "server_model_version": 8,
  "round_triggered": false,
  "queued_clients": 1
}
```

`server_model_version` only advances when `round_triggered` is `true` — i.e. when this upload completed a batch of `CENTRALIZED_CLIENTS_PER_ROUND` unique clients and a training round ran.

**GET response (200):**
```json
{
  "version": 8,
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
| `GET` | `/dev/metrics/json` | FL aggregation state snapshot |
| `WS`  | `/dev/metrics/ws` | Live metrics stream (pushes every 2 s) |
| `GET` | `/dev/metrics` | HTML dashboard |
| `GET` | `/dev/db/snapshot` | Database snapshot as JSON response |
| `GET` | `/dev/db/snapshot/export` | Download database snapshot (`.json` or `.json.gz` with `compress=true`) |

---

## Snapshot and backup strategy

- Use `/api/v1/dev/db/snapshot` and `/api/v1/dev/db/snapshot/export` for developer inspection and local analysis.
- For large payloads, use compressed export:

```bash
curl "http://localhost:8000/api/v1/dev/db/snapshot/export?compress=true" -o db_snapshot.json.gz
```

## Federated learning

FedAvg is triggered when **exactly** `FEDERATED_CLIENTS_PER_ROUND` unique clients have uploaded (default `2`). There is no timeout — the round waits indefinitely until the required number of clients arrives. This gives strict batch semantics that match the centralized arm's training batch size, so both experimental arms receive updates with the same amount of client signal.

Uploads are keyed by `client_id`, so re-uploads from the same client replace the previous entry rather than growing the queue. Aggregated backbone versions are persisted to PostgreSQL in the `federated_model_versions` table. Per-round metrics (clients, interactions, duration, model size) are logged in `aggregation_events`.

---

## Centralized training

Uploads are **buffered** in memory and keyed by `client_id`. A training round runs when **exactly** `CENTRALIZED_CLIENTS_PER_ROUND` unique clients have uploaded (default `2`). Re-uploads from the same client before the round triggers append more tuples to that client's buffer but do not increase the unique-client count. There is no timeout.

This mirrors the federated arm's strict batch semantics: both arms receive updates with the same number of client contributions per round, so their comparison is fair. Set `CENTRALIZED_CLIENTS_PER_ROUND` equal to `FEDERATED_CLIENTS_PER_ROUND`.

When a round triggers, the service:

1. Merges the round's buffered tuples into the persistent pool (a sliding window bounded by `MAX_TUPLE_POOL_SIZE`, default `2000`).
2. Retrains the backbone on the full pool (3 epochs, Adam lr=1e-3).
3. Applies a Bayesian online update to the global item, price, and nudge heads for each tuple contributed in this round.
4. Increments `model_version` and persists backbone + heads + pool to PostgreSQL in the `centralized_model_versions` table.
5. Logs round-level training metrics (duration, interactions, clients, losses, resource usage, model size) in `centralized_training_events`.
6. Clears the in-memory round buffer.

The persistent pool is fully restored on server restart. The in-memory round buffer is not — partially-filled batches at restart time are re-collected from the next client uploads (same liveness guarantee as the FL aggregator's queue).

---

## Project structure

```
.
├── app/
│   ├── main.py                    # ASGI entry point
│   ├── logger.py
│   ├── api/
│   │   ├── app.py                 # App factory + lifespan (aggregator + centralized service init)
│   │   ├── routers/
│   │   │   ├── api.py             # /api/v1 root router
│   │   │   ├── health.py
│   │   │   ├── catalogue.py
│   │   │   ├── images.py
│   │   │   ├── federated.py       # Federated backbone endpoints
│   │   │   ├── centralized.py     # Centralized training endpoints
│   │   │   └── metrics.py         # Dev dashboard
│   │   ├── schemas/               # Pydantic request/response models
│   │   │   ├── federated.py
│   │   │   ├── centralized.py
│   │   │   ├── catalogue_snapshot.py
│   │   │   ├── catalogue_version.py
│   │   │   ├── food_item.py
│   │   │   ├── category.py
│   │   │   ├── substitution_group.py
│   │   │   ├── substitution_group_item.py
│   │   │   ├── food_item_category.py
│   │   │   ├── product_label_image.py
│   │   │   └── common.py
│   │   └── services/
│   │       └── catalogue_snapshot.py
│   ├── backbones/
│   │   ├── aggregator.py          # FedAvg aggregation service
│   │   └── centralized.py         # Centralized backbone, heads, retraining, persistence
│   ├── db/
│   │   ├── database.py            # Async SQLAlchemy engine + session
│   │   ├── models/
│   │   │   ├── federated.py       # FederatedModel
│   │   │   ├── centralized.py     # CentralizedModel
│   │   │   ├── catalogue_version.py
│   │   │   ├── category.py
│   │   │   ├── food_item.py
│   │   │   ├── food_item_category.py
│   │   │   ├── substitution_group.py
│   │   │   └── substitution_group_item.py
│   │   └── seeding/               # Bootstrap + seed package (python -m app.db.seeding)
│   │       ├── runner.py          # ensure_models / seed_all / bootstrap_if_empty
│   │       ├── seed_backbone.py   # Pretrained backbone seed (version 1)
│   │       ├── seed_catalogue.py  # Food catalogue seed
│   │       ├── seed_status.py     # has_tables / is_database_seeded
│   │       ├── data/              # Catalogue CSV/JSON + pretrained/ weights
│   │       └── __main__.py        # CLI entrypoint
│   ├── web/                       # Dev dashboard static assets
│   ├── static/                    # Food item / product label images
│   └── requirements.txt
├── pretrain/                      # Offline backbone pretraining utilities
├── .github/workflows/ci.yml       # Lint + build/push to ghcr.io
├── scripts/
│   └── pg_dump_backup.sh          # DB-native backup script (pg_dump custom format)
├── Dockerfile.server              # Multi-stage server image
├── Dockerfile.pretrain            # Pretrain job image
├── docker-compose.dev.yml         # Local Postgres + bind-mounted source
├── docker-compose.prod.yml        # GHCR image + external RDS
└── .env.example
```

---

## Environment variables

| Variable | Default | Scope | Description |
|----------|---------|-------|-------------|
| `POSTGRES_USER` | `fedrl` | dev | Database user (dev compose) |
| `POSTGRES_PASSWORD` | `fedrl` | dev | Database password (dev compose) |
| `POSTGRES_DB` | `fedrl` | dev | Database name (dev compose) |
| `POSTGRES_HOST` | `localhost` | dev | Host for local tooling (not used by the server in-container) |
| `POSTGRES_PORT` | `5432` | dev | Port exposed by the local Postgres container |
| `DATABASE_URL` | — | prod | Full async SQLAlchemy URL (`postgresql+asyncpg://…`). Required in prod; overrides the `POSTGRES_*` vars. |
| `SQL_ECHO` | `false` | both | Log all SQL queries (dev only) |
| `CORS_ALLOW_ORIGINS` | `*` | both | Comma-separated allowed origins |
| `UVICORN_WORKERS` | `1` | both | Keep at `1` — round queues live in process memory |
| `FEDERATED_CLIENTS_PER_ROUND` | `2` | both | Unique client uploads required per FedAvg round |
| `CENTRALIZED_CLIENTS_PER_ROUND` | `2` | both | Unique client uploads required per centralized round (match `FEDERATED_CLIENTS_PER_ROUND`) |
| `MAX_TUPLE_POOL_SIZE` | `2000` | both | Sliding-window cap on the centralized tuple pool |
| `PRETRAINED_WEIGHTS_PATH` | `app/db/seeding/data/pretrained/pretrained_backbone_weights.npz` | seed | Source `.npz` for the v1 backbone seed |

---

## CI/CD

Every push to `main` runs Ruff and builds + pushes a Docker image to GHCR:

```
ghcr.io/magnusquist/fedrl-recommendation-server:latest
ghcr.io/magnusquist/fedrl-recommendation-server:sha-<commit>
```

Pull requests run lint and build-only (no push) to validate the Dockerfile.

---

## Development notes

- **No authentication** — the server assumes a trusted local network between Pi clients and the developer's machine. By design for the thesis MVP.
- **One server, three modes** — a single process handles federated and centralized clients simultaneously. Control-mode clients do not contact the server at all.
- **Experimental fairness** — both arms are seeded from identical initial backbone weights (the same pretrained `.npz` produced by `pretrain/`) so comparisons are valid from step 0.
- **Symmetric batch semantics across arms** — both federated and centralized rounds trigger on exactly N unique client uploads (`FEDERATED_CLIENTS_PER_ROUND` / `CENTRALIZED_CLIENTS_PER_ROUND`). Keep both values equal so the two arms train on the same amount of client signal per round.
- **No Alembic** — the schema is managed by `Base.metadata.create_all()` at seed time. Adding a column to an existing database requires a manual `ALTER TABLE` or wiping the volume (dev) / RDS snapshot + replace (prod).
- **Round metrics are event-sourced** — model version tables (`federated_model_versions`, `centralized_model_versions`) store only model state needed for the next round. Per-round metrics are recorded in `aggregation_events` and `centralized_training_events`.
