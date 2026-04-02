# FedRL — Recommendation Server

Federated RL recommendation server for the **Nudge2Green** thesis project.  
Exposes a food catalogue REST API and federated learning aggregation endpoints consumed by Raspberry Pi clients.

---

## Stack

| Layer | Technology |
|-------|------------|
| API framework | FastAPI + Uvicorn |
| Database | PostgreSQL 16 |
| ORM / schema | SQLAlchemy 2 (async); `create_all` for new tables |
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

On startup, the server runs **`Base.metadata.create_all()`** when `AUTO_CREATE_SCHEMA=true` (default in Docker Compose). That **creates missing tables** from the ORM in `app/api/schemas/`. It does **not** rename columns, drop columns, or change types on existing tables.

For a fresh Postgres volume you need nothing extra. If you change models against an old database, either apply manual SQL, or reset the volume / truncate and recreate.

### 4. Seed the food catalogue

The catalogue is seeded from three JSON files under `data/` by [`app/db/seed_catalogue.py`](app/db/seed_catalogue.py):

| File | Role |
|------|------|
| `data/categories.json` | Hierarchical taxonomy (main + `sub_categories`, stable numeric ids). |
| `data/substitution_groups.json` | Substitution groups (`id`, `name`, `product_ids`). |
| `data/product_items.json` | Product rows joined to subcategories and groups. |
| `data/product_items.jsonl` | *(optional)* One JSON object per line; if present, seed **streams** this file instead of loading the whole `.json` into memory. Generate with `python data/convert_catalogue_to_jsonl.py`. |

**Progress:** Catalogue seed logs scan/insert progress every 2000 rows.

```bash
# Optional: build JSON Lines from the large array (faster / lower memory on seed)
python data/convert_catalogue_to_jsonl.py

# From project root with venv + DATABASE_URL set, or inside the server container:
python -m app.db.seed_catalogue
# e.g. Docker Compose:
docker compose exec server python -m app.db.seed_catalogue
```

On first run (empty `food_items` table), the script inserts categories with explicit ids and `parent_id`, substitution groups with explicit ids, then products in batches. It advances PostgreSQL sequences after those inserts so future autoincrement rows do not collide.

**Switching catalogues:** truncate `food_item_substitution_groups`, `food_items`, `substitution_groups`, and `categories` (or use a fresh database volume), restart the server (or run `seed_catalogue`), then seed again.

Startup behaviour is in [`app/api/helpers/seed_db.py`](app/api/helpers/seed_db.py): `AUTO_CREATE_SCHEMA=true` runs `create_all` before the server listens; `AUTO_SEED_DATA_ON_STARTUP=true` (default) runs backbone + catalogue seed **in the background**. Set `AUTO_SEED_DATA_ON_STARTUP=false` if you seed manually.

### 5. Verify

```bash
curl http://localhost:8000/health
# → {"status": "ok", "database": "reachable"}

curl http://localhost:8000/catalogue
# → {"version": "...", "item_count": 0, "items": []}
```

Interactive API docs: **http://localhost:8000/docs**

---

## Changing the schema

The database schema is defined by **SQLAlchemy ORM** classes under `app/api/schemas/`. There is **no migration runner**: startup uses `create_all`, which only creates **missing** tables.

**Adding a new table:** add the model, restart with `AUTO_CREATE_SCHEMA=true` — the new table appears.

**Changing columns on an existing table:** `create_all` will **not** alter it. For a thesis / dev setup, common options are:

- **Wipe and recreate** — `docker compose down -v` (removes the Postgres volume), then `up` again; or run targeted `ALTER TABLE` in psql / pgAdmin.
- **Use a migration workflow** — e.g. numbered `.sql` scripts or an external migration tool — if you need versioned upgrades later.

Always mirror API types in `app/api/models/` (Pydantic) when you change ORM columns.

**Reseed after catalogue shape changes:** update `app/db/seed_catalogue.py` and/or `data/*.json`, truncate catalogue tables (or fresh volume), then:

```bash
docker compose exec server python -m app.db.seed_catalogue
```

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
- **Catalogue seeding** — after `create_all`, run `python -m app.db.seed_catalogue` (or enable `AUTO_SEED_DATA_ON_STARTUP`) so `data/categories.json`, `substitution_groups.json`, and `product_items.json` populate the database.