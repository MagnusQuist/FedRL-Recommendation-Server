# FedRL Recommendation Server

Backend for the Nudge2Green federated RL thesis project. Exposes a food catalogue API and endpoints for federated backbone aggregation and centralized training.

## Setup

```bash
cp .env.example .env
# Edit .env as needed
docker compose -f docker-compose.dev.yml up --build
```

The API is available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## Database

The database schema is created and seeded automatically on startup — there are no migrations. Every restart assumes a clean rebuild; if the schema changes, drop and recreate the database.

## Production

```bash
# Requires DATABASE_URL pointing to an external Postgres instance (e.g. Amazon RDS)
docker compose -f docker-compose.prod.yml up
```

See [aws-deployment.md](./aws-deployment.md) for AWS-specific setup.
