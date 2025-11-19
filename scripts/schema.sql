-- TimescaleDB Schema for Swing Trading System

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Core price table (hypertable on time)
CREATE TABLE IF NOT EXISTS prices (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    vwap NUMERIC,
    PRIMARY KEY (time, ticker)
);

-- Convert to hypertable
SELECT create_hypertable('prices', 'time', if_not_exists => TRUE);

-- Create index on ticker
CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);

-- Feature table
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    -- Technical
    rsi_14 NUMERIC,
    macd NUMERIC,
    macd_signal NUMERIC,
    bbands_upper NUMERIC,
    bbands_lower NUMERIC,
    atr_14 NUMERIC,
    volume_z_score NUMERIC,
    -- Cross-sectional
    return_rank_5d INTEGER,
    return_rank_20d INTEGER,
    sector_relative_strength NUMERIC,
    correlation_to_spy NUMERIC,
    -- Text (from LLM)
    sentiment_score NUMERIC,
    news_intensity VARCHAR(20),
    event_flag_earnings BOOLEAN,
    -- Pattern flags
    breakout_52w BOOLEAN,
    pattern_confidence NUMERIC,
    pattern_type VARCHAR(50),
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_ticker ON features(ticker);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    model_version VARCHAR(20),
    expected_return NUMERIC,
    hit_prob_long NUMERIC,
    hit_prob_short NUMERIC,
    volatility_forecast NUMERIC,
    quantile_10 NUMERIC,
    quantile_50 NUMERIC,
    quantile_90 NUMERIC,
    priority_score NUMERIC,
    PRIMARY KEY (time, ticker, model_version)
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker);

-- Recommendations table (final output)
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10), -- 'buy' or 'sell'
    target_pct NUMERIC,
    stop_pct NUMERIC,
    probability NUMERIC,
    priority_score NUMERIC,
    position_size_pct NUMERIC,
    rationale TEXT[],
    model_version VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_recommendations_date ON recommendations(date);
CREATE INDEX IF NOT EXISTS idx_recommendations_ticker ON recommendations(ticker);

-- Performance tracking
CREATE TABLE IF NOT EXISTS performance (
    id SERIAL PRIMARY KEY,
    recommendation_id INTEGER REFERENCES recommendations(id),
    entry_date DATE,
    exit_date DATE,
    entry_price NUMERIC,
    exit_price NUMERIC,
    actual_return NUMERIC,
    hit_target BOOLEAN,
    days_held INTEGER,
    exit_reason VARCHAR(50) -- 'target', 'stop', 'time', 'manual'
);

CREATE INDEX IF NOT EXISTS idx_performance_entry_date ON performance(entry_date);

-- Paper trading table
CREATE TABLE IF NOT EXISTS paper_trades (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    side VARCHAR(10), -- 'buy' or 'sell'
    entry_price NUMERIC,
    target_price NUMERIC,
    stop_price NUMERIC,
    exit_price NUMERIC,
    exit_date DATE,
    pnl_pct NUMERIC,
    predicted_probability NUMERIC,
    status VARCHAR(20) DEFAULT 'open', -- 'open' or 'closed'
    exit_reason VARCHAR(50), -- 'target', 'stop', 'time', 'manual'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_date ON paper_trades(date);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(status);
CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker);

-- Stock characteristics table (updated weekly)
CREATE TABLE IF NOT EXISTS stock_characteristics (
    ticker VARCHAR(10) PRIMARY KEY,
    sector VARCHAR(20),
    market_cap BIGINT,
    beta NUMERIC,
    avg_volume_20d BIGINT,
    avg_dollar_volume_20d NUMERIC,
    mean_reversion_strength NUMERIC,  -- Hurst exponent
    earnings_sensitivity NUMERIC,  -- Avg % move on earnings
    sentiment_beta NUMERIC,  -- Sensitivity to sentiment
    liquidity_regime INT,  -- 0=low, 1=mid, 2=high
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stock_characteristics_sector ON stock_characteristics(sector);
CREATE INDEX IF NOT EXISTS idx_stock_characteristics_updated ON stock_characteristics(updated_at);

-- Twin predictions table (per stock per day)
CREATE TABLE IF NOT EXISTS twin_predictions (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    twin_version VARCHAR(20),
    expected_return NUMERIC,
    hit_prob NUMERIC,
    volatility NUMERIC,
    quantile_10 NUMERIC,
    quantile_50 NUMERIC,
    quantile_90 NUMERIC,
    regime VARCHAR(20),  -- Trending/MeanReverting/Choppy/Volatile
    idiosyncratic_alpha NUMERIC,  -- Twin-specific adjustment
    PRIMARY KEY (time, ticker, twin_version)
);

SELECT create_hypertable('twin_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_twin_predictions_ticker ON twin_predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_twin_predictions_date_ticker ON twin_predictions(time, ticker);

-- Options price data (raw)
CREATE TABLE IF NOT EXISTS options_prices (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    strike_price NUMERIC NOT NULL,
    expiration DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL, -- 'call' or 'put'
    
    -- Pricing
    bid NUMERIC,
    ask NUMERIC,
    last_price NUMERIC,
    
    -- Greeks
    delta NUMERIC,
    gamma NUMERIC,
    vega NUMERIC,
    theta NUMERIC,
    rho NUMERIC,
    
    -- Volume & OI
    volume INTEGER,
    open_interest INTEGER,
    
    -- IV
    implied_volatility NUMERIC,
    
    PRIMARY KEY (time, ticker, strike_price, expiration, option_type)
);

SELECT create_hypertable('options_prices', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_options_prices_ticker ON options_prices(ticker);
CREATE INDEX IF NOT EXISTS idx_options_prices_expiration ON options_prices(expiration);

-- Options features (40 features per stock per day)
CREATE TABLE IF NOT EXISTS options_features (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    
    -- Volume & OI (8 features)
    call_oi BIGINT,
    put_oi BIGINT,
    total_oi BIGINT,
    call_volume BIGINT,
    put_volume BIGINT,
    total_volume BIGINT,
    oi_change_pct NUMERIC,
    volume_zscore NUMERIC,
    
    -- Put-Call Ratios (6 features)
    pcr_oi NUMERIC,
    pcr_volume NUMERIC,
    pcr_zscore NUMERIC,
    pcr_extreme_bullish BOOLEAN,
    pcr_extreme_bearish BOOLEAN,
    pcr_change NUMERIC,
    
    -- Gamma Exposure (7 features)
    max_pain_strike NUMERIC,
    max_pain_distance_pct NUMERIC,
    total_gamma NUMERIC,
    gamma_sign INTEGER,
    gamma_concentration NUMERIC,
    gamma_flip_strike NUMERIC,
    gamma_flip_distance_pct NUMERIC,
    
    -- Implied Volatility (7 features)
    atm_call_iv NUMERIC,
    atm_put_iv NUMERIC,
    iv_skew NUMERIC,
    put_call_iv_ratio NUMERIC,
    iv_percentile NUMERIC,
    iv_rank NUMERIC,
    iv_change_pct NUMERIC,
    
    -- Net Greeks (5 features)
    net_delta NUMERIC,
    net_gamma NUMERIC,
    net_vega NUMERIC,
    net_theta NUMERIC,
    net_delta_abs NUMERIC,
    
    -- Term Structure (4 features)
    front_month_oi BIGINT,
    next_month_oi BIGINT,
    roll_ratio NUMERIC,
    term_curve_slope NUMERIC,
    
    -- Composite Signals (3 features)
    trend_signal NUMERIC,
    sentiment_signal NUMERIC,
    gamma_signal INTEGER,
    
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('options_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_options_features_ticker ON options_features(ticker);
CREATE INDEX IF NOT EXISTS idx_options_features_date_ticker ON options_features(time, ticker);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_recommendations_date_priority ON recommendations(date, priority_score DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_date_ticker ON predictions(time, ticker);
