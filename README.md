# FedRL — Recommendation Server

Federated RL recommendation server for the **Nudge2Green** thesis project.  
Exposes a food catalogue REST API and learning aggregation endpoints for Raspberry Pi clients running LinTS (Linear Thompson Sampling) in one of three experiment modes.

---

## Experiment modes

| Mode | Description | Server involvement |
|------|-------------|-------------------|
| **federated** | Clients retrain locally and upload backbone weights; server aggregates via FedAvg | `GET/POST /api/v1/federated/model` |
| **centralized** | Clients upload raw interaction tuples; server trains the model centrally | `GET /api/v1/centralized/model`, `POST /api/v1/centralized/interactions` |
| **control** | No model, no server communication | None |

Both `federated` and `centralized` are initialised from the **same pretrained backbone weights** (produced by the pretrain pipeline) so the experiment starts from an identical baseline. A single server instance handles all client modes simultaneously.

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

### 3. Database schema & seeding

Bootstrap is handled by a one-shot `seed` service in docker-compose that runs `python -m app.db.seeding` (see [`app/db/seeding`](app/db/seeding/)) before the `server` service starts. When the target database has no tables it:

1. Creates the schema via `Base.metadata.create_all()`.
2. Seeds the federated + centralized backbones from `PRETRAINED_WEIGHTS_PATH` (defaults to `app/db/seeding/data/pretrained/pretrained_backbone_weights.npz`).
3. Seeds the food catalogue from the JSON/CSV files under `app/db/seeding/data/`.

If the schema already exists the seed container is a no-op and exits immediately, so it is safe on every `docker compose up`. For schema changes on an existing database, apply manual SQL or wipe the volume.

### 4. Re-seeding manually

```bash
# Force re-create tables + re-run every seeder (full replace):
docker compose run --rm seed python -m app.db.seeding --force

# Re-seed only parts of the database:
docker compose run --rm seed python -m app.db.seeding --catalogue-only
docker compose run --rm seed python -m app.db.seeding --backbones-only
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
| `GET` | `/api/v1/federated/model?since=<v>&algorithm=ts` | Download global backbone if newer than `since`; returns 304 otherwise |
| `POST` | `/api/v1/federated/model` | Upload local backbone weights for FedAvg aggregation |
| `GET` | `/api/v1/federated/version?algorithm=ts` | Current global backbone version |
| `GET` | `/api/v1/federated/status?algorithm=ts` | Aggregation queue status (queued clients, rounds completed, timeout) |

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

FedAvg is triggered when **exactly** `FL_MIN_CLIENTS_PER_ROUND` unique clients have uploaded (default `2`). There is no timeout — the round waits indefinitely until the required number of clients arrives. This gives strict batch semantics that match the centralized arm's training batch size, so both experimental arms receive updates with the same amount of client signal.

Uploads are keyed by `client_id`, so re-uploads from the same client replace the previous entry rather than growing the queue. Aggregated backbone versions are persisted to PostgreSQL (`federated_backbone_versions` table).

---

## Centralized training

Uploads are **buffered** in memory and keyed by `client_id`. A training round runs when **exactly** `CENTRALIZED_CLIENTS_PER_ROUND` unique clients have uploaded (default `2`). Re-uploads from the same client before the round triggers append more tuples to that client's buffer but do not increase the unique-client count. There is no timeout — the round waits indefinitely until the required number of clients arrives.

This mirrors the federated arm's strict batch semantics: both arms receive updates with the same number of client contributions per round, so their comparison is fair. Set `CENTRALIZED_CLIENTS_PER_ROUND` equal to `FEDERATED_CLIENTS_PER_ROUND`.

When a round triggers, the service:

1. Merges the round's buffered tuples into the persistent pool (a sliding window bounded by `MAX_TUPLE_POOL_SIZE`, default `2000`).
2. Retrains the backbone on the full pool (3 epochs, Adam lr=1e-3).
3. Applies a Bayesian online update to the global item, price, and nudge heads for each tuple contributed in this round.
4. Increments `model_version` and persists backbone + heads + pool to PostgreSQL (`centralized_model_versions` table).
5. Clears the in-memory round buffer.

The persistent pool is fully restored on server restart. The in-memory round buffer is not — partially-filled batches at restart time are re-collected from the next client uploads (same liveness guarantee as the FL aggregator's queue).

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
│   │       ├── federated.py       # Federated request/response models
│   │       └── centralized.py     # Centralized request/response models
│   ├── db/
│   │   ├── database.py            # Async SQLAlchemy engine + session
│   │   ├── models/
│   │   │   └── federated.py       # FederatedBackboneVersion ORM model
│   │   └── seeding/               # Bootstrap + seed package (python -m app.db.seeding)
│   │       ├── runner.py          # ensure_models / seed_all / bootstrap_if_empty
│   │       ├── seed_backbone.py   # Pretrained backbone seed (version 1)
│   │       ├── seed_catalogue.py  # Food catalogue seed
│   │       ├── seed_status.py     # has_tables / is_database_seeded helpers
│   │       ├── data/              # Catalogue CSV/JSON + pretrained/ weights
│   │       └── __main__.py        # CLI entrypoint
│   ├── fl/
│   │   ├── aggregator.py          # FedAvg aggregation service
│   │   └── centralized.py        # Centralized backbone, heads, retraining, persistence
│   ├── web/                       # Dev dashboard static assets
│   └── requirements.txt
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
| `FEDERATED_CLIENTS_PER_ROUND` | `2` | Exact number of unique client uploads required per FedAvg round |
| `CENTRALIZED_CLIENTS_PER_ROUND` | `2` | Exact number of unique client uploads required per centralized training round (should match `FEDERATED_CLIENTS_PER_ROUND`) |

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
- **Experimental fairness** — both federated and centralized backbones are seeded from identical initial weights (the same pretrained `npz` produced by the pretrain pipeline) so comparisons are valid from step 0.
- **Symmetric batch semantics across arms** — both federated and centralized rounds trigger on exactly N unique client uploads (`FEDERATED_CLIENTS_PER_ROUND` / `CENTRALIZED_CLIENTS_PER_ROUND`). Keep both values equal so the two arms train on the same amount of client signal per round.
