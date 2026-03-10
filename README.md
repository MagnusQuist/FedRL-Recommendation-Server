# FedRL — Recommendation Server

Federated RL recommendation server for the **Nudge2Green** thesis project.  
Exposes a food catalogue REST API and federated learning aggregation endpoints consumed by Raspberry Pi clients.

---

## Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| ORM / migrations | SQLAlchemy 2 (async) + Alembic |
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

### 3. Run database migrations

```bash
# From the project root, with PostgreSQL running:
DATABASE_URL=postgresql+asyncpg://fedrl:fedrl@localhost:5432/fedrl \
  alembic upgrade head
```

Or run migrations inside the container:

```bash
docker compose exec server alembic upgrade head
```

### 4. Seed the food catalogue

The catalogue is seeded from `data/seed_catalogue.json` via `app/seed_catalogue.py`.

```bash
# With Docker Compose running:
docker compose exec server python -m app.seed_catalogue
```

The script:
- Reads `data/seed_catalogue.json` (structure: `{ "items": [ ... ] }`).
- Maps each item to the `FoodItem` ORM model in `app/schemas/food_item.py`:
  - `id` → `item_id`
  - `name` → `name`
  - `category` → `category`
  - `total_co2e` → `co2e_emission_tonnes`
  - `price` → `price`
  - `greener_alternative_ids` → `alternative_ids` (resolved to UUIDs in a second pass).

### 5. Verify

```bash
curl http://localhost:8000/health
# → {"status": "ok", "database": "reachable"}

curl http://localhost:8000/catalogue
# → {"version": "...", "item_count": 0, "items": []}
```

Interactive API docs: **http://localhost:8000/docs**

---

## Changing the schema safely

The database schema is owned by SQLAlchemy models (`app/schemas/food_item.py`) + Alembic migrations (`alembic/versions/`).

**Typical workflow to change the schema:**

1. **Edit the ORM model**
   - Change `FoodItem` (or other models) in `app/schemas/food_item.py` — add/remove/rename columns, adjust types, etc.

2. **Update Pydantic models**
   - Mirror the changes in `app/models/food_item.py` so API responses match the DB.

3. **Create a new migration (on the host)**
   - With a local virtualenv:

   ```bash
   source .venv/bin/activate
   export DATABASE_URL=postgresql+asyncpg://fedrl:fedrl@localhost:5432/fedrl
   alembic revision -m "Describe your change" --autogenerate
   ```

   - Review the generated file under `alembic/versions/` and tweak if needed.

4. **Apply migrations**

   ```bash
   alembic upgrade head
   # or, after rebuilding the image so migrations are copied in:
   # docker compose exec server alembic upgrade head
   ```

5. **Reseed if necessary**
   - If the change affects the catalogue shape, update `app/seed_catalogue.py` to map the JSON fields to the new columns, then rerun:

   ```bash
   docker compose exec server python -m app.seed_catalogue
   ```

This keeps the code, migrations, and running database in sync while keeping seeding a single, repeatable command.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server + database status |
| `GET` | `/catalogue` | Full food catalogue |
| `GET` | `/catalogue/category/{name}` | Items filtered by category |
| `GET` | `/catalogue/{id}` | Single item by UUID |
| `POST` | `/fl/upload` | *(stub)* Upload backbone weights |
| `GET` | `/fl/model` | *(stub)* Download global backbone |

---

## Project structure

```
.
├── app/
│   ├── main.py          # FastAPI app + lifespan
│   ├── database.py      # SQLAlchemy async engine + session
│   ├── models/          # ORM models (FoodItem, GlobalBackboneVersion)
│   ├── schemas/         # Pydantic request/response schemas
│   └── routers/         # health, catalogue, fl (stub)
├── alembic/                 # Database migrations
├── .github/workflows/       # CI/CD — build & push to ghcr.io
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

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

- **No authentication** — the server assumes a trusted local network between Pi clients and the developer's machine. This is by design for the thesis MVP.
- **FL aggregation** — the `/fl/upload` and `/fl/model` endpoints are stubs. Aggregation logic will be added in a subsequent implementation sprint.
- **Catalogue seeding** — the food catalogue is empty after a fresh migration. A seed script will be added alongside the catalogue data.