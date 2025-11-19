# Local Setup Guide

This guide covers setting up the Swing Trading System to run entirely on your local MacBook, including databases, storage, and inference.

## Architecture Overview

The system is designed to run locally with the following components:

- **Databases**: TimescaleDB, Neo4j, Redis (via Docker Compose)
- **Storage**: Local filesystem (replaced S3)
- **Inference**: Local MacBook
- **Training**: Google Colab (optional, see [COLAB_TRAINING.md](./COLAB_TRAINING.md))

## Prerequisites

- macOS (tested on macOS 13+)
- Docker Desktop installed and running
- Python 3.11+ (tested with Python 3.11.13)
- Node.js 16+ (for UI)

## Step 1: Install Dependencies

```bash
# Clone repository (if not already done)
cd /path/to/swing

# Install Python dependencies
pip install -r requirements.txt
```

**Note**: 
- TimescaleDB does not require a Python package (it's a PostgreSQL extension). The `psycopg2-binary` package (already in requirements) is sufficient for connecting to TimescaleDB.
- The codebase uses `talib` (TA-Lib) for technical analysis, not `pandas-ta`. The `pandas-ta` package is not included in requirements.

## Step 2: Configure Environment

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit config.yaml with your settings
# - API keys (Polygon, Finnhub, OpenAI, FRED)
# - Database passwords
# - Options data configuration (if using options layer)
```

Key configuration sections in `config/config.yaml`:

```yaml
api_keys:
  polygon:
    api_key: "YOUR_POLYGON_API_KEY"  # Required for options data (Starter plan or higher)

data_sources:
  options:
    enabled: false  # Set to true to enable options data ingestion
    provider: "polygon"
    tickers: []  # Empty = use all tickers, or specify subset (e.g., top 100 liquid stocks)
    max_contracts_per_ticker: 1000

storage:
  local:
    data_dir: "data/"          # Local data directory
    models_dir: "models/"      # Local models directory
  
  timescaledb:
    host: "localhost"
    port: 5432
    database: "swing_trading"
    user: "postgres"
    password: "YOUR_PASSWORD"
  
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "YOUR_NEO4J_PASSWORD"
  
  feast:
    registry_path: "data/feast/registry.db"  # Local Feast registry

models:
  twins:
    options_enabled: false  # Set to true to enable options encoder in twins
    options_embedding_dim: 32

rl_portfolio:
  options_enabled: false  # Set to true to enable options signals in RL agent
```

## Step 3: Start Local Databases

The system uses Docker Compose to run three databases locally:

### Start All Databases

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

This starts:
- **TimescaleDB** on `localhost:5432` (PostgreSQL with time-series extensions)
- **Neo4j** on `localhost:7687` (Bolt) and `localhost:7474` (HTTP)
- **Redis** on `localhost:6379` (for Feast online store)

### Verify Database Connections

```bash
# Test TimescaleDB
psql -h localhost -U postgres -d swing_trading -c "SELECT version();"

# Test Neo4j (open in browser)
open http://localhost:7474

# Test Redis
redis-cli -h localhost -p 6379 ping
```

### Initialize Database Schema

**Note**: The schema is automatically initialized when TimescaleDB starts for the first time. The `schema.sql` file is mounted to `/docker-entrypoint-initdb.d/` and will run automatically.

If you need to manually reinitialize the schema:

```bash
# Using docker exec
docker exec -i swing-timescaledb psql -U postgres -d swing_trading < scripts/schema.sql

# Or using local psql (if installed)
psql -h localhost -U postgres -d swing_trading < scripts/schema.sql
```

## Step 4: Set Up Local Storage

The system uses local filesystem storage instead of S3. Create the directory structure:

```bash
# Create data directories
mkdir -p data/raw/prices
mkdir -p data/raw/options  # Options chain data (if enabled)
mkdir -p data/raw/news
mkdir -p data/raw/fundamentals
mkdir -p data/raw/macro
mkdir -p data/processed/features
mkdir -p data/processed/features/options  # Processed options features
mkdir -p data/feast

# Create models directories
mkdir -p models/foundation
mkdir -p models/twins
```

Or let the system create them automatically on first use.

## Step 5: Start Services

### Start API Server

```bash
# From project root
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Start React Dashboard (Optional)

```bash
cd src/ui
npm install
npm start
```

The UI will be available at `http://localhost:3000`

## Step 6: Run Data Ingestion

Ingest initial data to populate the databases:

```bash
# Using Prefect (if configured)
prefect deployment run data-ingestion/daily

# Or run directly
python -m src.data.ingestion
```

This will:
- Fetch market data from Polygon API
- Fetch options data from Polygon (if enabled)
- Fetch news from Finnhub
- Fetch fundamentals and macro data
- Save to local filesystem (`data/raw/`)
- Save prices to TimescaleDB
- Save options prices and features to TimescaleDB (if enabled)

## Step 7: Verify Setup

### Check Data Storage

```bash
# List ingested data
ls -lh data/raw/prices/
ls -lh data/raw/options/  # If options enabled
ls -lh data/raw/news/

# Check database records
psql -h localhost -U postgres -d swing_trading -c "SELECT COUNT(*) FROM prices;"
psql -h localhost -U postgres -d swing_trading -c "SELECT COUNT(*) FROM options_prices;"  # If options enabled
psql -h localhost -U postgres -d swing_trading -c "SELECT COUNT(*) FROM options_features;"  # If options enabled
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/

# Get recommendations
curl http://localhost:8000/api/recommendations
```

## Local Storage Structure

```
swing/
├── data/                          # Local data storage
│   ├── raw/                       # Raw ingested data
│   │   ├── prices/                # Price data (Parquet)
│   │   │   └── YYYY-MM-DD/
│   │   ├── options/               # Options chain data (JSON/Parquet, if enabled)
│   │   │   └── YYYY-MM-DD/
│   │   ├── news/                  # News data (JSON)
│   │   ├── fundamentals/          # Fundamentals (Parquet)
│   │   └── macro/                 # Macro data (JSON)
│   ├── processed/                 # Processed features
│   │   └── features/
│   │       └── options/           # Processed options features (if enabled)
│   └── feast/                     # Feast feature store
│       └── registry.db
│
├── models/                        # Model checkpoints
│   ├── foundation/                # Foundation model
│   └── twins/                     # Digital twin models
│       └── {TICKER}/
│
└── config/
    └── config.yaml                # Configuration
```

## Database Management

### Backup Databases

```bash
# Backup TimescaleDB
docker exec swing-timescaledb pg_dump -U postgres swing_trading > backup_$(date +%Y%m%d).sql

# Backup Neo4j (requires Neo4j tools)
docker exec swing-neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j.dump
```

### Stop Databases

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f timescaledb
docker-compose logs -f neo4j
docker-compose logs -f redis
```

## Troubleshooting

### Database Connection Issues

**Problem**: Cannot connect to TimescaleDB
```bash
# Check if container is running
docker ps | grep timescaledb

# Check logs
docker-compose logs timescaledb

# Restart service
docker-compose restart timescaledb
```

**Problem**: Neo4j authentication fails
- Default password is `changeme` (set via `NEO4J_PASSWORD` env var)
- Change password in Neo4j browser: `http://localhost:7474`
- Update `config.yaml` with new password

### Storage Issues

**Problem**: Permission denied when saving data
```bash
# Check directory permissions
ls -la data/

# Fix permissions
chmod -R 755 data/
```

**Problem**: Disk space issues
- Monitor disk usage: `du -sh data/`
- Clean old data: Remove files older than X days
- Compress old data: Use Parquet compression

### Port Conflicts

If ports are already in use:

```yaml
# Edit docker-compose.yml to use different ports
services:
  timescaledb:
    ports:
      - "5433:5432"  # Use 5433 instead of 5432
```

Then update `config.yaml` accordingly.

## Performance Optimization

### TimescaleDB Tuning

For better performance with large datasets:

```sql
-- Enable compression
ALTER TABLE prices SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'ticker'
);

-- Create continuous aggregates
CREATE MATERIALIZED VIEW prices_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS hour,
       ticker,
       AVG(close) AS avg_close,
       MAX(high) AS max_high,
       MIN(low) AS min_low
FROM prices
GROUP BY hour, ticker;
```

### Neo4j Memory Settings

Edit `docker-compose.yml` to increase memory:

```yaml
neo4j:
  environment:
    NEO4J_dbms_memory_heap_initial__size: 2g
    NEO4J_dbms_memory_heap_max__size: 4g
```

## Next Steps

- **Training**: See [COLAB_TRAINING.md](./COLAB_TRAINING.md) for training models in Google Colab
- **Architecture**: See [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) for system architecture
- **Twin Training**: See [TWIN_TRAINING.md](./TWIN_TRAINING.md) for digital twin training procedures

## Environment Variables

You can override config settings with environment variables:

```bash
export TIMESCALEDB_PASSWORD=your_password
export NEO4J_PASSWORD=your_password
export POLYGON_API_KEY=your_key
export FINNHUB_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

These are used by `docker-compose.yml` and can override `config.yaml` settings.

