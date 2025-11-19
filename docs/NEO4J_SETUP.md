# Neo4j Setup and Usage

## Overview

Neo4j is used to store and query the weighted correlation graph between stocks. This graph is computed daily from rolling price correlations and persisted to Neo4j for use by the GNN model.

## Architecture

- **Nodes**: `Stock` nodes with `ticker` and `sector` properties
- **Edges**: `CORRELATES` relationships with:
  - `weight`: Correlation coefficient (-1 to 1)
  - `abs_weight`: Absolute value of correlation
  - `date`: Date of the correlation calculation
  - `lookback_days`: Number of days used for correlation (typically 30)

## Setup

### Using Docker Compose

Neo4j is included in `docker-compose.yml`. To start:

```bash
docker-compose up -d neo4j
```

The service will be available at:
- HTTP: http://localhost:7474
- Bolt: bolt://localhost:7687

Default credentials:
- Username: `neo4j`
- Password: Set via `NEO4J_PASSWORD` environment variable (default: `changeme`)

### Manual Setup

1. Download Neo4j Community Edition from https://neo4j.com/download/
2. Install and start Neo4j
3. Set initial password
4. Update `config/config.yaml` with connection details

## Configuration

Add to `config/config.yaml`:

```yaml
storage:
  neo4j:
    uri: "bolt://localhost:7687"
    user: "neo4j"
    password: "YOUR_PASSWORD"
    database: "neo4j"
```

Or set environment variables:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

## Usage in Pipeline

### Graph Persistence

During feature engineering (Step 3.5), the pipeline:
1. Computes correlation matrix from price data (30-day rolling window)
2. Creates edges for correlations above threshold (default: 0.3)
3. Persists nodes and edges to Neo4j with weighted relationships

### Graph Retrieval

During model inference (Step 5), the pipeline:
1. Queries Neo4j for correlation graph for the current date
2. Filters edges by correlation threshold
3. Converts to PyTorch Geometric format for GNN

## Cypher Queries

### View all stocks
```cypher
MATCH (s:Stock)
RETURN s.ticker, s.sector
LIMIT 100
```

### View correlations for a specific date
```cypher
MATCH (s1:Stock)-[r:CORRELATES {date: '2025-11-18'}]->(s2:Stock)
WHERE r.abs_weight >= 0.3
RETURN s1.ticker, s2.ticker, r.weight
ORDER BY r.abs_weight DESC
LIMIT 50
```

### Find most correlated stocks to a ticker
```cypher
MATCH (s1:Stock {ticker: 'AAPL'})-[r:CORRELATES]->(s2:Stock)
WHERE r.date = '2025-11-18'
RETURN s2.ticker, r.weight
ORDER BY r.abs_weight DESC
LIMIT 10
```

### Get graph statistics
```cypher
MATCH ()-[r:CORRELATES {date: '2025-11-18'}]->()
RETURN 
  count(r) as total_edges,
  avg(r.abs_weight) as avg_abs_correlation,
  min(r.weight) as min_correlation,
  max(r.weight) as max_correlation
```

## API

The Neo4j client (`src/data/neo4j_client.py`) provides:

- `upsert_stock_node(ticker, sector)`: Create/update stock node
- `upsert_correlation_edge(ticker1, ticker2, correlation, date)`: Create/update edge
- `batch_upsert_correlations(correlations, date)`: Batch upsert
- `get_correlation_graph(date, tickers, threshold)`: Retrieve graph
- `get_latest_correlation_graph(tickers, threshold)`: Get most recent graph

## Graph Construction Functions

- `build_correlation_graph()`: Compute and persist to Neo4j
- `build_correlation_graph_from_neo4j()`: Load from Neo4j for GNN

## Performance Considerations

- **Batch Operations**: Use `batch_upsert_correlations()` for efficiency
- **Indexing**: Neo4j automatically indexes node properties
- **Threshold Filtering**: Filter by `abs_weight >= threshold` in queries
- **Date Filtering**: Always filter by date to get correct graph snapshot

## Troubleshooting

### Connection Issues
- Verify Neo4j is running: `docker ps | grep neo4j`
- Check connection string in config
- Verify credentials

### No Data
- Check if pipeline has run and persisted graphs
- Verify date format matches (YYYY-MM-DD)
- Check Neo4j logs: `docker logs swing-neo4j`

### Performance
- Ensure Neo4j has sufficient memory allocated
- Consider creating indexes on date if querying large datasets
- Use parameterized queries (already implemented in client)

