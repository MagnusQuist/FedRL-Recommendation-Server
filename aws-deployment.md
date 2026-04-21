# Development Guide for FedRL-Server Deployment on AWS

## Part 1: Security Groups

### Step 1 (fedrl-server-sg)

Console → EC2 → Security Groups → Create security group:
    1. Name: fedrl-server-sg
    2. Description: FedRL server EC2 - SSH + API
    3. VPC: default VPC (the one labelled (default))
    4. Inbound rules:
        - Type: SSH, Source: My IP (auto-fills your public IP with a /32 mask)
        - Type: Custom TCP, Port range: 8000, Source: fedrl-clients-sg after creation of it in " Step 2 ".
    5. Outbound rules: leave default (All traffic to 0.0.0.0/0)
    6. Click Create security group.

### Step 2 (fedrl-clients-sg)

Same flow:
    1. Name: fedrl-clients-sg
    2. Description: FedRL clients EC2 - SSH only inbound
    3. VPC: default
    4. Inbound rules:
        - SSH: 22 from My IP.
        - Custom TCP: 443 from 0.0.0.0/0
        - Custom TCP: 80 from 0.0.0.0/0
    5. Outbound rules: default

### Step 3 (fedrl-db-sg)

Same flow again:
    1. Name: fedrl-db-sg
    2. Description: FedRL Postgres - only from fedrl-server-sg
    3. VPC: default
    4. Inbound rules:
        - Type: PostgreSQL (auto-fills port 5432), Source: Custom → start typing sg- and select fedrl-server-sg.
    5. Outbound rules: default

### Step 4 (Add the cross-SG rule to fedrl-server-sg)

Now that fedrl-clients-sg exists, edit fedrl-server-sg and add an inbound rule:
    - Type: Custom TCP, Port: 8000, Source: fedrl-clients-sg (select it from the dropdown).

## Part 2: Create the RDS instance

Console → RDS → Create database:
    1. Choose Standard create.
    2. Engine: PostgreSQL, version 17.x (match your docker-compose.dev.yml: postgres:17.6).
    3. Templates: Free tier (or Production if you have budget).
    4. Settings (* = obscured in this documents):
        - DB instance identifier: fedrl-db-****
        - Master username: fedrl-****
        - Master password: ********
    5. DB instance class: db.t4g.micro (free tier).
    6. Storage: 20 GB gp2, disable autoscaling for predictability. (free tier)
        - Note: Might want to upgrade... → gp3 is alot better and cheaper but is not free tier.
    7. Connectivity:
        - VPC: default VPC.
        - DB subnet group: default.
        - Public access: No.
        - VPC security group: Choose existing → remove the default → add fedrl-db-sg.
        - Availability Zone: any single one (eu-north-1a is fine).
        - Port: 5432.
    8. Additional configuration (* = obscured in this documents):
        - Initial database name: postgres_fedrl_*****.
        - Backup retention: 1 day (Might want this to be the full duration of the experiment).
        - Deletion protection: off for now; on later if you don't want to accidentally nuke it.

## Part 3: Launch and provision the server EC2

### Step 1 (Launch EC2)

Console → EC2 → Launch instance:
    1. Name: fedrl-server
    2. AMI: Ubuntu Server 24.04 LTS (free tier eligible).
    3. Instance type: t3.micro (free tier) or t3.small (more headroom).
    4. Key pair: fedrl-key.pem (This is important for ssh)
    5. Network settings → Edit:
        - VPC: default.
        - Subnet: any default subnet (pick one in eu-north-1a to match RDS for lowest latency).
        - Auto-assign public IP: Enable.
        - Firewall → Select existing security group → fedrl-server-sg.
    6. Storage: 20 GB gp2.
        - Note: Might want to upgrade... → gp3 is alot better and cheaper but is not free tier.
    7. Launch

### Step 2 (SSH in and install Docker)

```bash
ssh -i fedrl-key.pem ubuntu@<public-ip>
```

When inside the fedrl-server EC2:

```bash
# System updates
sudo apt-get update && sudo apt-get upgrade -y
# Docker (official install script)
curl -fsSL https://get.docker.com | sudo sh
# Let ubuntu user run docker without sudo
sudo usermod -aG docker ubuntu
# Log out and back in so the group takes effect
exit
```

SSH back in:

```bash
docker version          # should show client + server versions
docker compose version  # should show Docker Compose version
```

### Step 3 (Get the compose file and .env onto the EC2)

On your laptop:

```bash
scp -i fedrl-key.pem docker-compose.prod.yml ubuntu@<public-ip>:~/
```

Create .env on the EC2 (in the same location as the docker-compose.prod.yml file):

```bash
touch .env
```

### Step 4 (Configure .env on the EC2)

```bash
sudo nano .env
```

Paste the variables (Update so they are correct):

```
SERVER_IMAGE=ghcr.io/magnusquist/fedrl-recommendation-server:latest
DATABASE_URL=postgresql+asyncpg://fedrl:<PASSWORD>@fedrl-db.xxxxx.eu-north-1.rds.amazonaws.com:5432/postgres_fedrl
SQL_ECHO=false
CORS_ALLOW_ORIGINS=*
UVICORN_WORKERS=1
FEDERATED_CLIENTS_PER_ROUND=2
CENTRALIZED_CLIENTS_PER_ROUND=2
MAX_TUPLE_POOL_SIZE=2000
```

### Step 5 (Bring the server online and check it's running)

```bash
docker compose -f docker-compose.prod.yml up -d

# Smoke test
curl http://localhost:8000/api/v1/health
# {"status":"ok","database":"reachable"}
```

