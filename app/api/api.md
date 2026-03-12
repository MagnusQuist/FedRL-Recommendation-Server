# API Endpoints
Definition of system endpoints

### Catalogue

Base URL: `localhost/api/v1/catalogue`

| Method | Endpoint       | Description                  |
|--------|----------------|------------------------------|
| **GET**    | `/`         | Retrieve all food items           |
| **GET**     | `/version`         | Current catalogue version          |
| **GET**     | `/snapshot`         | Combined; food items, categories, substitution_groups, catalogue version          |
| **GET**     | `/categories`         | Retrieve all categories          |
| **GET**     | `/categories/{id}`         | Retrieve items in a category by ID          |
| **GET**     | `/item/{id}`         | Retrieve specific food item          |
| **GET**     | `/substitution_groups`         | Get all substitution groups          |
| **GET**     | `/substitution_groups/item?item_id={id}`         | Get an item's substitution group and related items          |

### Backbone

Base URL `localhost/api/v1/backbone`

| Method | Endpoint       | Description                  |
|--------|----------------|------------------------------|
| **GET**    | `/status`         | Aggregation queue status           |
| **GET**    | `/version`         | Global backbone version           |
| **GET**     | `/model`         | Get current global backbone model          |
| **POST**     | `/model`         | Upload backbone weights for FedAvg aggregation        |
