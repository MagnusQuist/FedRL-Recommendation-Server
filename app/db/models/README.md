# Database Models Reference

This folder defines the SQLAlchemy ORM models for the server.
The tables are grouped into two domains:

- **Training/model state** (federated + centralized)
- **Catalogue domain** (food items, categories, substitution groups)

## Training and Model Tables

### `federated_model_versions`
Stores the current/previous federated backbone snapshots.

- `id` (PK, int)
- `version` (unique int)
- `weights_blob` (gzip + base64 JSON backbone weights)
- `created_at` (UTC timestamp)

### `centralized_model_versions`
Stores full centralized model snapshots needed to continue centralized training.

- `id` (PK, int)
- `version` (unique int)
- `backbone_blob`
- `reward_predictor_blob`
- `item_head_blob`
- `price_head_blob`
- `nudge_head_blob`
- `tuple_pool_blob` (serialized interaction pool state)
- `created_at` (UTC timestamp)

### `aggregation_events`
Round-level log table for each completed FedAvg aggregation.

- `aggregation_event_id` (PK, int)
- `timestamp` (aggregation start time)
- `aggregation_duration_ms`
- `participating_clients_ids` (PostgreSQL `ARRAY(TEXT)`)
- `num_clients_in_round`
- `total_interactions`
- `model_version_before`
- `model_version_after`
- `model_size_bytes`
- `logged_at`

Index:
- `idx_aggregation_events_timestamp` on `timestamp`

### `centralized_training_events`
Round-level log table for each completed centralized training round.

- `centralized_training_event_id` (PK, int)
- `timestamp` (round start time)
- `training_duration_ms`
- `num_interactions`
- `num_clients_contributing`
- `contributing_client_ids` (PostgreSQL `ARRAY(TEXT)`)
- `cpu_usage_percentage`
- `memory_usage_mb`
- `loss_before`
- `loss_after`
- `loss_delta`
- `model_version_before`
- `model_version_after`
- `model_size_bytes`
- `logged_at`

Index:
- `idx_centralized_training_events_timestamp` on `timestamp`

## Catalogue Tables

### `catalogue_versions`
Tracks generated catalogue snapshots/versions.

- `id` (PK, int)
- `version` (unique string)
- `generated_at`

### `categories`
Master table for item categories.

- `category_id` (PK, int)
- `name` (unique)
- `slug` (unique)

### `food_items`
Main product table with nutrition, sustainability, flags, and price metadata.

Primary key:
- `id` (string product ID)

Key fields include:
- identity: `name`, `brand`
- product facts: `product_weight_in_g`
- nutrition: calories/macros/salt fields
- sustainability: CO2-related fields and labels
- booleans (e.g. gluten-free/frozen/fairtrade/etc.)
- pricing: `price_dkk`

Indexes:
- `ix_food_items_name`
- `ix_food_items_brand`

### `food_item_categories`
Many-to-many join table between `food_items` and `categories`.

Composite PK:
- `product_id` -> `food_items.id`
- `category_id` -> `categories.category_id`

Indexes:
- `ix_food_item_categories_product_id`
- `ix_food_item_categories_category_id`

### `substitution_groups`
Groups of interchangeable food items.

- `substitution_group_id` (PK, int)
- `name` (unique)

### `substitution_group_items`
Many-to-many join table between `substitution_groups` and `food_items`.

Composite PK:
- `substitution_group_id` -> `substitution_groups.substitution_group_id`
- `product_id` -> `food_items.id`

Indexes:
- `ix_substitution_group_items_group_id`
- `ix_substitution_group_items_product_id`

## Relationship Overview

- `food_items` <-> `categories` via `food_item_categories`
- `food_items` <-> `substitution_groups` via `substitution_group_items`
- Model state is in:
  - `federated_model_versions`
  - `centralized_model_versions`
- Round/event metrics are in:
  - `aggregation_events`
  - `centralized_training_events`
