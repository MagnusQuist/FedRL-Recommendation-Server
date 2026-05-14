# API Endpoints

All endpoints are prefixed with `/api/v1`.

## Health

Base URL: `http://localhost:8000/api/v1`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Server and database status. |
| `GET` | `/seed-status` | Whether the database contains seeded data. |

### `GET /health`

Response:

```json
{
  "status": "ok",
  "database": "reachable"
}
```

### `GET /seed-status`

Response:

```json
{
  "seeded": true
}
```

## Catalogue

Base URL: `http://localhost:8000/api/v1/catalogue`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/snapshot` | Full catalogue snapshot with items, categories, substitution groups, and version. |
| `GET` | `/version` | Current catalogue version. |

## Images

Base URL: `http://localhost:8000/api/v1/images`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/food_item/{food_item_id}` | Serve a food item WebP image. |
| `GET` | `/product_label/{label_name}` | Serve a product label WebP image. |

## Federated

Base URL: `http://localhost:8000/api/v1/federated`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/status` | Federated aggregation queue status. |
| `GET` | `/version` | Current federated backbone version. |
| `GET` | `/model?since={version}` | Download current federated backbone, or `304` if the client is current. |
| `POST` | `/model` | Upload local backbone weights for aggregation. |

### `GET /status`

Response:

```json
{
  "current_version": 3,
  "queued_clients": ["client_01"],
  "total_rounds_completed": 2,
  "clients_per_round": 2
}
```

### `GET /version`

Response:

```json
{
  "version": 3
}
```

### `GET /model?since={version}`

Returns `304 Not Modified` when `since` is greater than or equal to the current stored version.

Response `200`:

```json
{
  "version": 3,
  "client_count": 2,
  "total_interactions": 40,
  "backbone_weights": "<gzip-base64-json>"
}
```

`backbone_weights` is a gzip-compressed, base64-encoded JSON object containing the backbone state dict.

### `POST /model`

Request:

```json
{
  "client_id": "client_01",
  "backbone_version": 3,
  "interaction_count": 10,
  "backbone_weights": "<gzip-base64-json>"
}
```

`backbone_weights` must contain exactly these keys:

```text
backbone.0.weight
backbone.0.bias
backbone.2.weight
backbone.2.bias
```

Response `202`:

```json
{
  "status": "queued",
  "client_id": "client_01",
  "queued_clients": 1,
  "round_triggered": false
}
```

Round behavior:

- uploads are keyed by `client_id`
- a repeat upload from the same client replaces that client's queued upload
- a round triggers when `queued_clients == FEDERATED_CLIENTS_PER_ROUND`
- default `FEDERATED_CLIENTS_PER_ROUND` is `2`

## Centralized

Base URL: `http://localhost:8000/api/v1/centralized`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/status` | Centralized training queue status. |
| `GET` | `/version` | Current centralized model version. |
| `GET` | `/model?since={version}` | Download current centralized model, or `304` if the client is current. |
| `POST` | `/interactions` | Upload interaction tuples for centralized training. |

### `GET /status`

Response:

```json
{
  "current_version": 8,
  "queued_clients": 1,
  "total_rounds_completed": 7,
  "pool_size": 280,
  "clients_per_round": 2
}
```

### `GET /version`

Response:

```json
{
  "version": 8
}
```

### `GET /model?since={version}`

Returns `304 Not Modified` when `since` is greater than or equal to the current service version.

Response `200`:

```json
{
  "version": 8,
  "backbone_weights": "<gzip-base64-json>",
  "reward_predictor_weights": "<gzip-base64-json>",
  "head_weights": {
    "item": "<gzip-base64-json>",
    "price": "<gzip-base64-json>",
    "nudge": "<gzip-base64-json>"
  }
}
```

### `POST /interactions`

Request:

```json
{
  "client_id": "client_01",
  "count": 10,
  "data": "<gzip-base64-json-array>"
}
```

`data` must decode to a JSON array of interaction tuples. Each tuple must contain a `context` vector with the server model's expected length.

Response `200`:

```json
{
  "accepted": true,
  "server_model_version": 8,
  "round_triggered": false,
  "queued_clients": 1
}
```

Round behavior:

- uploads are keyed by `client_id`
- a repeat upload from the same client replaces that client's queued tuples
- a round triggers when `queued_clients == CENTRALIZED_CLIENTS_PER_ROUND`
- default `CENTRALIZED_CLIENTS_PER_ROUND` is `2`
- the tuple pool is capped by `MAX_TUPLE_POOL_SIZE`, default `2000`

## Development

Base URL: `http://localhost:8000/api/v1/dev/db`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/snapshot/json` | JSON snapshot of key database tables. |
| `GET` | `/snapshot/export` | Downloadable JSON snapshot. |

Query parameters:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_rows_per_table` | integer | `null` | Optional row cap per table, from `1` to `10000`. |
| `include_model_blobs` | boolean | `true` | Include model blob columns in the snapshot. |
| `compress` | boolean | `false` | Only for `/snapshot/export`; returns `.json.gz` when true. |
