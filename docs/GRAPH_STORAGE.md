# Graph Storage Documentation

## Overview

The system uses **parquet-based storage** for correlation graphs instead of a graph database. Graphs are computed daily from price data and cached to parquet files for fast retrieval.

## Architecture

### Storage Location

Graphs are stored in `data/graphs/` directory:

```
data/graphs/
├── correlations/
│   └── correlations_YYYY-MM-DD.parquet  # Daily correlation edges
└── metadata/
    └── metadata_YYYY-MM-DD.json         # Graph construction metadata
```

### File Format

#### Correlation Parquet File

Columns:
- `ticker1`: First stock ticker
- `ticker2`: Second stock ticker  
- `correlation`: Correlation coefficient (-1 to 1)
- `abs_correlation`: Absolute value of correlation

Only edges with `abs_correlation >= threshold` (default 0.3) are stored.

#### Metadata JSON File

```json
{
  "date": "2024-01-15",
  "lookback_days": 30,
  "threshold": 0.3,
  "num_edges": 1250,
  "num_nodes": 500,
  "computation_time": "2024-01-15T10:30:00"
}
```

## Usage

### Building and Caching Graphs

```python
from src.features.graph import build_correlation_graph

# Build graph and cache to parquet
graph, ticker_to_idx = build_correlation_graph(
    prices_df,
    date="2024-01-15",
    threshold=0.3,
    cache_to_parquet=True,
    config=config
)
```

### Loading from Cache

```python
from src.features.graph import build_correlation_graph_from_parquet

# Load from parquet cache
graph, ticker_to_idx = build_correlation_graph_from_parquet(
    date="2024-01-15",
    tickers=None,  # Optional: filter to specific tickers
    threshold=0.3,
    config=config
)
```

### Automatic Cache Management

The `get_or_build_graph()` function automatically:
1. Checks for cached graph in parquet
2. Loads if found
3. Computes fresh if missing
4. Caches the result

```python
from src.features.graph_storage import get_or_build_graph

correlations, ticker_to_idx = get_or_build_graph(
    prices_df,
    date="2024-01-15",
    threshold=0.3,
    lookback_days=30,
    config=config,
    force_recompute=False  # Set True to recompute even if cache exists
)
```

## Benefits

1. **No Database Server**: No need to run Neo4j or any graph database
2. **Fast I/O**: Parquet files are optimized for columnar reads
3. **Version Control Friendly**: Small parquet files can be committed to git
4. **Easy Inspection**: Can open parquet files with pandas/polars for analysis
5. **Simple Backup**: Just copy `data/graphs/` directory
6. **Computation on Demand**: Graphs computed fresh if cache missing

## Daily Pipeline Integration

The daily pipeline automatically:
1. Computes correlation matrix from price data (30-day rolling window)
2. Filters edges by correlation threshold (default 0.3)
3. Saves to `data/graphs/correlations/YYYY-MM-DD.parquet`
4. Saves metadata to `data/graphs/metadata/YYYY-MM-DD.json`

During inference:
1. Attempts to load graph from parquet cache
2. Falls back to computation if cache missing
3. Uses PyTorch Geometric format for GNN processing

## Migration from Neo4j

If you have existing Neo4j data, you can export it:

```python
# Export from Neo4j (one-time migration)
from src.data.neo4j_client import Neo4jClient

neo4j = Neo4jClient(...)
edges, ticker_to_idx = neo4j.get_correlation_graph(date, ...)

# Save to parquet
from src.features.graph_storage import save_correlation_graph

correlations = {(t1, t2): w for t1, t2, w in edges}
save_correlation_graph(correlations, date, ticker_to_idx, ...)
```

However, since graphs are computed fresh daily, migration is typically not needed.

## Configuration

No special configuration needed. Graph storage directory is determined by:

```yaml
storage:
  local:
    data_dir: "data/"  # Graphs stored in data/graphs/
```

## Performance

- **Computation**: ~1-2 seconds for 500 stocks (30-day correlation)
- **Parquet Write**: ~100ms for typical graph
- **Parquet Read**: ~50ms for cached graph
- **Total Cache Hit**: ~50ms (vs ~1-2s computation)

## Troubleshooting

**Problem**: Graph files not found
- Graphs are computed automatically during pipeline execution
- Check `data/graphs/correlations/` directory
- Run pipeline once to generate graphs

**Problem**: Out of date graphs
- Graphs are computed fresh daily
- Use `force_recompute=True` to regenerate
- Old graphs are not automatically cleaned (can be deleted manually)

**Problem**: Large file sizes
- Typical graph: ~100-500KB for 500 stocks
- If files are large, consider increasing correlation threshold
- Old graphs can be archived/deleted

