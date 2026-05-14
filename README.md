# FedRL Recommendation Server

Federated and centralized recommendation server for the Nudge2Green thesis project. The server exposes a food catalogue API, static product images, federated backbone aggregation, and centralized training endpoints under one FastAPI app.

[AWS deployment guide](./aws-deployment.md)

## Experiment Modes

| Mode | Client behavior | Server behavior |
| --- | --- | --- |
| `federated` | Clients train locally and upload backbone weights. | Aggregates uploaded backbones with FedBuff and serves the latest global backbone. |
| `centralized` | Clients upload raw interaction tuples. | Retrains a centralized backbone, reward predictor, and global heads. |
| `control` | No model upload/download. | No training endpoint involvement. |

Federated and centralized model rows are seeded from the same pretrained backbone weights so both training arms start from the same baseline.

## Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Database | PostgreSQL, SQLAlchemy 2 async |
| ML | PyTorch for centralized training, NumPy for FedBuff aggregation |
| Packaging | Docker Compose, GHCR image build in CI |

## Quick Start

```bash
git clone https://github.com/MagnusQuist/FedRL-Recommendation-Server.git
cd FedRL-Recommendation-Server
cp .env.example .env
docker compose -f docker-compose.dev.yml up --build
```

The API is served on `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

## Database Seeding

The seed package is `app.db.seed`.

```bash
python -m app.db.seed
python -m app.db.seed --force
python -m app.db.seed --catalogue-only
python -m app.db.seed --backbones-only
```

Seeding creates missing tables via `Base.metadata.create_all()`, then inserts:

- catalogue data from `app/db/seed/data/`
- federated backbone version `1`
- centralized model version `1`, including backbone, reward predictor, empty heads, and an empty tuple pool

The project does not use Alembic. Schema changes on an existing database require a manual migration or recreating the database.

## API Summary

All endpoints are mounted under `/api/v1`. See [docs/api.md](./docs/api.md) for request and response details.

| Area | Endpoints |
| --- | --- |
| Health | `GET /health`, `GET /seed-status` |
| Catalogue | `GET /catalogue/snapshot`, `GET /catalogue/version` |
| Images | `GET /images/food_item/{food_item_id}`, `GET /images/product_label/{label_name}` |
| Federated | `GET /federated/status`, `GET /federated/version`, `GET /federated/model`, `POST /federated/model` |
| Centralized | `GET /centralized/status`, `GET /centralized/version`, `GET /centralized/model`, `POST /centralized/interactions` |
| Development | `GET /dev/db/snapshot/json`, `GET /dev/db/snapshot/export` |

## Federated Learning

Federated rounds trigger when exactly `FEDERATED_CLIENTS_PER_ROUND` unique clients have queued uploads. The default is `2`.

Uploads are keyed by `client_id`; if the same client uploads again before a round triggers, the new upload replaces the old one. When the queue reaches the required number of unique clients, the server:

1. Loads the current global backbone.
2. Loads each upload's base backbone version.
3. Drops uploads whose base version is unavailable.
4. Applies FedBuff using `interaction_count` and staleness discounting.
5. Persists a new row in `federated_model_versions`.
6. Logs round metadata in `aggregation_events`.
7. Clears the in-memory queue.

FedBuff settings:

| Variable | Default | Description |
| --- | --- | --- |
| `FEDERATED_CLIENTS_PER_ROUND` | `2` | Unique clients required to trigger a round. |
| `FEDERATED_SERVER_LR` | `1.0` | Server learning rate applied to averaged deltas. |
| `FEDERATED_STALENESS_ALPHA` | `0.5` | Staleness discount exponent. |

## Centralized Training

Centralized rounds trigger when exactly `CENTRALIZED_CLIENTS_PER_ROUND` unique clients have queued interaction uploads. The default is `2`.

Uploads are keyed by `client_id`; a re-upload from the same client before a round triggers replaces that client's queued tuples. There is no timeout.

When a round triggers, the server:

1. Collects queued tuples and appends them to a sliding tuple pool capped by `MAX_TUPLE_POOL_SIZE`.
2. Updates the global item, price, and nudge heads using the new round tuples.
3. Retrains the centralized backbone and reward predictor on the full tuple pool.
4. Increments the centralized model version.
5. Persists backbone, reward predictor, heads, and tuple pool in `centralized_model_versions`.
6. Logs round metrics in `centralized_training_events`.
7. Clears the in-memory queue.

Training defaults:

| Setting | Value |
| --- | --- |
| Backbone LR | `1e-4` |
| Reward predictor LR | `1e-3` |
| Epochs | `5` |
| Batch size | `64` |
| Gradient clip | `1.0` |
| Tuple pool cap | `MAX_TUPLE_POOL_SIZE`, default `2000` |

## Project Structure

```text
app/
  api/
    app.py                 # FastAPI app factory and lifespan state
    routes.py              # /api/v1 router
    endpoints/             # health, catalogue, images, federated, centralized, dev
  db/
    models.py              # SQLAlchemy models
    session.py             # async engine/session
    seed/                  # python -m app.db.seed
  ml/
    federated/
      aggregation.py       # federated queue and round orchestration
      backbone_codec.py    # backbone blob encoding/decoding
      fedbuff.py           # pure FedBuff math
    centralized/
      training.py          # centralized queue, round orchestration, persistence
      trainer.py           # pure training/evaluation functions
      models.py            # BackboneEncoder, RewardPredictor, optimizer
      heads.py             # Thompson-sampling heads
      codec.py             # centralized blob encoding/decoding
  schemas/                 # Pydantic schemas
  services/                # catalogue and snapshot services
  static/                  # food item and product label images
docker/
  server.Dockerfile
  pretrain.Dockerfile
docs/
  api.md
tests/
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DATABASE_URL` | derived from `POSTGRES_*` in local use | Async SQLAlchemy URL. Required for production. |
| `POSTGRES_USER` | `fedrl` | Local Postgres user. |
| `POSTGRES_PASSWORD` | `fedrl` | Local Postgres password. |
| `POSTGRES_DB` | `fedrl` | Local Postgres database. |
| `POSTGRES_HOST` | `localhost` | Local Postgres host for tooling. |
| `POSTGRES_PORT` | `5432` | Local Postgres port. |
| `SQL_ECHO` | `false` | Log SQL statements. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins. |
| `UVICORN_WORKERS` | `1` | Keep at `1`; training queues are in process memory. |
| `FEDERATED_CLIENTS_PER_ROUND` | `2` | Unique federated clients per round. |
| `FEDERATED_SERVER_LR` | `1.0` | FedBuff server learning rate. |
| `FEDERATED_STALENESS_ALPHA` | `0.5` | FedBuff staleness discount exponent. |
| `CENTRALIZED_CLIENTS_PER_ROUND` | `2` | Unique centralized clients per round. |
| `MAX_TUPLE_POOL_SIZE` | `2000` | Centralized tuple pool sliding-window size. |
| `PRETRAINED_WEIGHTS_PATH` | `app/db/seed/data/pretrained/pretrained_backbone_weights.npz` | Optional v1 backbone seed weights. |

## Development Checks

```bash
python -m unittest discover -s tests
python -m ruff check app tests
```

## Notes

- There is no authentication; the thesis setup assumes trusted clients.
- Keep `UVICORN_WORKERS=1` because queues live in process memory.
- Keep federated and centralized client-per-round values aligned when comparing experimental arms.
- Round metrics are event rows; model version tables store the state needed to serve and continue training.
