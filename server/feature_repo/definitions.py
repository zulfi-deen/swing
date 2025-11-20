from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

# Define the entity
ticker = Entity(name="ticker", join_keys=["ticker"])

# Sources (pointing to parquet files in S3/local)
technical_stats_source = FileSource(
    name="technical_stats_source",
    path="/app/data/technical_features.parquet",  # Path in Docker container
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Feature Views
technical_features = FeatureView(
    name="technical_features",
    entities=[ticker],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rsi_14", dtype=Float32),
        Field(name="macd", dtype=Float32),
        Field(name="macd_signal", dtype=Float32),
        Field(name="bbands_pct", dtype=Float32),
        Field(name="atr_14", dtype=Float32),
        Field(name="volume_z_score", dtype=Float32),
        Field(name="sentiment_score", dtype=Float32),
    ],
    online=True,
    source=technical_stats_source,
    tags={"team": "swing_trading"},
)

