# API Endpoints
Definition of system endpoints

All endpoints are prefixed with `/api/v1`.

---

### Health

Base URL: `localhost/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/health` | Server and database status |
| **GET** | `/seed-status` | Whether the database has been seeded |

---

### Catalogue

Base URL: `localhost/api/v1/catalogue`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/snapshot` | Combined snapshot — food items, categories, substitution groups, and catalogue version |
| **GET** | `/version` | Current catalogue version |

---

### Images

Base URL: `localhost/api/v1/images`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/food_item/{food_item_id}` | Serve food item image (webp) |
| **GET** | `/product_label/{label_name}` | Serve product label image (webp) |

---

### Federated

Base URL: `localhost/api/v1/federated`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/status` | Aggregation queue status |
| **GET** | `/version` | Federated backbone version |
| **GET** | `/model?since={version}` | Download current federated backbone; returns 304 if client is already up to date |
| **POST** | `/model` | Upload backbone weights for FedAvg aggregation |

---

### Centralized

Base URL: `localhost/api/v1/centralized`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/status` | Centralized training queue status |
| **GET** | `/version` | Centralized model version |
| **GET** | `/model?since={version}` | Download current centralized model (backbone + heads); returns 304 if client is already up to date |
| **POST** | `/interactions` | Upload interaction tuples for centralized training |

---

### Development

Base URL: `localhost/api/v1/dev/db`

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/snapshot/json` | JSON snapshot of key database tables. Query params: `max_rows_per_table`, `include_model_blobs` |
| **GET** | `/snapshot/export` | Downloadable JSON snapshot. Query params: `max_rows_per_table`, `include_model_blobs`, `compress` |
