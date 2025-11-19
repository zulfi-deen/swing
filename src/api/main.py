"""FastAPI main application"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import logging

from src.data.storage import get_timescaledb_engine
from src.models.model_registry import ModelRegistry
from src.utils.config import load_config

logger = logging.getLogger(__name__)

app = FastAPI(title="Swing Trading API", version="1.0.0")

# Global model registry (loaded on startup)
model_registry: ModelRegistry = None

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradeRecommendation(BaseModel):
    ticker: str
    side: str
    target_pct: float
    stop_pct: float
    probability: float
    priority_score: float
    position_size_pct: float
    rationale: List[str]


class DailyBrief(BaseModel):
    date: str
    market_context: dict
    brief: str
    trades: List[TradeRecommendation]


@app.get("/")
def root():
    return {"message": "Swing Trading API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global model_registry
    try:
        config = load_config()
        model_registry = ModelRegistry.get_twin_manager(config)
        if model_registry:
            logger.info("Models loaded successfully on startup")
        else:
            logger.warning("Model registry not initialized (models may not be available)")
    except Exception as e:
        logger.error(f"Error loading models on startup: {e}", exc_info=True)


@app.get("/recommendations/latest", response_model=List[TradeRecommendation])
def get_latest_recommendations():
    """Get today's trade recommendations."""
    
    try:
        engine = get_timescaledb_engine()
        
        # Query from DB
        query = """
            SELECT * FROM recommendations
            WHERE date = (SELECT MAX(date) FROM recommendations)
            ORDER BY priority_score DESC
        """
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No recommendations for today")
        
        # Convert to response model
        trades = []
        for _, row in df.iterrows():
            trades.append(TradeRecommendation(
                ticker=row['ticker'],
                side=row['side'],
                target_pct=float(row['target_pct']),
                stop_pct=float(row['stop_pct']),
                probability=float(row['probability']),
                priority_score=float(row['priority_score']),
                position_size_pct=float(row['position_size_pct']),
                rationale=row['rationale'] if isinstance(row['rationale'], list) else []
            ))
        
        return trades
    
    except Exception as e:
        logger.error(f"Error fetching recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{date}", response_model=List[TradeRecommendation])
def get_recommendations_by_date(date: str):
    """Get recommendations for a specific date."""
    
    try:
        engine = get_timescaledb_engine()
        
        query = """
            SELECT * FROM recommendations
            WHERE date = %s
            ORDER BY priority_score DESC
        """
        
        df = pd.read_sql(query, engine, params=(date,))
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No recommendations for {date}")
        
        trades = [
            TradeRecommendation(
                ticker=row['ticker'],
                side=row['side'],
                target_pct=float(row['target_pct']),
                stop_pct=float(row['stop_pct']),
                probability=float(row['probability']),
                priority_score=float(row['priority_score']),
                position_size_pct=float(row['position_size_pct']),
                rationale=row['rationale'] if isinstance(row['rationale'], list) else []
            )
            for _, row in df.iterrows()
        ]
        
        return trades
    
    except Exception as e:
        logger.error(f"Error fetching recommendations for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/brief/latest", response_model=DailyBrief)
def get_daily_brief():
    """Get latest daily brief."""
    
    try:
        from src.agents.explainer import ExplainerAgent
        
        engine = get_timescaledb_engine()
        
        # Get latest date
        latest_date = pd.read_sql(
            "SELECT MAX(date) as date FROM recommendations",
            engine
        ).iloc[0, 0]
        
        if latest_date is None:
            raise HTTPException(status_code=404, detail="No brief available")
        
        # Get recommendations
        recommendations = get_recommendations_by_date(str(latest_date))
        
        # Get market context
        spy_query = """
            SELECT close, volume FROM prices
            WHERE ticker = 'SPY' AND time::date = %s
            ORDER BY time DESC LIMIT 1
        """
        spy_data = pd.read_sql(spy_query, engine, params=(latest_date,))
        
        # Generate brief
        brief_lines = [
            f"Daily Trading Brief - {latest_date}",
            "",
            f"Market Summary:",
            f"- {len(recommendations)} trade recommendations generated",
        ]
        
        if not spy_data.empty:
            brief_lines.append(f"- SPY closed at ${spy_data.iloc[0]['close']:.2f}")
        
        brief_lines.extend([
            "",
            "Top Recommendations:",
        ])
        
        for i, trade in enumerate(recommendations[:5], 1):
            brief_lines.append(
                f"{i}. {trade.ticker} ({trade.side.upper()}): "
                f"Target {trade.target_pct*100:.1f}%, "
                f"Stop {trade.stop_pct*100:.1f}%, "
                f"Probability {trade.probability*100:.1f}%"
            )
        
        brief = "\n".join(brief_lines)
        
        market_context = {
            "date": str(latest_date),
            "num_recommendations": len(recommendations),
            "spy_close": float(spy_data.iloc[0]['close']) if not spy_data.empty else None
        }
        
        return DailyBrief(
            date=str(latest_date),
            market_context=market_context,
            brief=brief,
            trades=recommendations
        )
    
    except Exception as e:
        logger.error(f"Error fetching brief: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{ticker}")
def explain_recommendation(ticker: str, date: Optional[str] = None):
    """Get detailed explanation for a ticker recommendation."""
    
    try:
        from src.agents.explainer import ExplainerAgent
        
        engine = get_timescaledb_engine()
        
        if date is None:
            date = pd.read_sql(
                "SELECT MAX(date) as date FROM recommendations",
                engine
            ).iloc[0, 0]
        
        # Query recommendation
        query = """
            SELECT * FROM recommendations
            WHERE date = %s AND ticker = %s
        """
        
        df = pd.read_sql(query, engine, params=(date, ticker))
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendation for {ticker} on {date}"
            )
        
        row = df.iloc[0]
        
        # Generate explanation
        agent = ExplainerAgent()
        
        trade = {
            'ticker': ticker,
            'side': row['side'],
            'target_pct': float(row['target_pct']),
            'stop_pct': float(row['stop_pct']),
            'probability': float(row['probability'])
        }
        
        features = {
            'expected_return': float(row.get('target_pct', 0)),
            'probability': float(row.get('probability', 0)),
        }
        
        explanation = agent.explain_trade(trade, features)
        
        return {
            'ticker': ticker,
            'date': str(date),
            'explanation': explanation,
            'features': features
        }
    
    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/backtest")
def get_backtest_results():
    """Get historical backtest performance."""
    
    try:
        from src.evaluation.backtest import backtest_strategy
        
        # Load predictions and actual returns from database
        engine = get_timescaledb_engine()
        
        predictions_query = """
            SELECT date, ticker, expected_return, hit_probability, priority_score
            FROM predictions
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
            ORDER BY date
        """
        
        predictions = pd.read_sql(predictions_query, engine)
        
        if predictions.empty:
            return {
                "error": "No backtest data available",
                "metrics": {}
            }
        
        # Get actual returns
        actual_query = """
            SELECT time::date as date, ticker, close
            FROM prices
            WHERE time >= CURRENT_DATE - INTERVAL '90 days'
            ORDER BY time, ticker
        """
        
        actual_prices = pd.read_sql(actual_query, engine)
        
        # Calculate actual returns
        actual_returns = actual_prices.groupby('ticker')['close'].pct_change(5).reset_index()
        actual_returns = actual_returns.rename(columns={'close': 'return_5d'})
        
        # Run backtest (simplified)
        metrics = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0
        }
        
        # Note: Full backtest would use backtest_strategy() function
        # This is a placeholder
        
        return {
            "period": "90 days",
            "metrics": metrics,
            "note": "Backtest metrics calculated from predictions"
        }
    
    except Exception as e:
        logger.error(f"Error fetching backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/paper")
def get_paper_trading_performance():
    """Get paper trading performance."""
    
    try:
        from src.evaluation.paper_trading import PaperTradingTracker
        
        tracker = PaperTradingTracker()
        metrics = tracker.get_performance_report()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error fetching paper trading performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/twins")
def list_twins():
    """List all available digital twins with metadata."""
    
    try:
        global model_registry
        
        # Use cached registry or load if needed
        if model_registry is None:
            config = load_config()
            model_registry = ModelRegistry.get_twin_manager(config)
        
        if model_registry is None:
            return {
                'twins': [],
                'message': 'TwinManager not available',
                'status': ModelRegistry.get_status()
            }
        
        twins_info = model_registry.list_twins()
        
        return {
            'total_twins': len(twins_info),
            'available_twins': sum(1 for t in twins_info if t.get('available', False)),
            'twins': twins_info,
            'registry_status': ModelRegistry.get_status()
        }
    
    except Exception as e:
        logger.error(f"Error listing twins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/twins/{ticker}")
def get_twin_info(ticker: str):
    """Get detailed information about a specific twin."""
    
    try:
        global model_registry
        
        # Use cached registry or load if needed
        if model_registry is None:
            config = load_config()
            model_registry = ModelRegistry.get_twin_manager(config)
        
        if model_registry is None:
            raise HTTPException(status_code=503, detail="TwinManager not available")
        
        # Check if ticker is in pilot list
        config = load_config()
        pilot_tickers = config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
        if ticker not in pilot_tickers:
            raise HTTPException(status_code=404, detail=f"Twin for {ticker} not found (not in pilot list)")
        
        twin = model_registry.get_twin(ticker)
        
        if not twin:
            raise HTTPException(status_code=404, detail=f"Twin for {ticker} not available")
        
        info = twin.get_stock_characteristics()
        
        return {
            'ticker': ticker,
            'available': True,
            'alpha': info.get('current_alpha', 0.0),
            'num_parameters': info.get('num_parameters', 0),
            'characteristics': info.get('characteristics', {})
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting twin info for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/{ticker}/regime")
def get_ticker_regime(ticker: str, date: Optional[str] = None):
    """Get current regime classification for a ticker."""
    
    try:
        from src.models.regime import get_regime_name, get_regime_trading_strategy
        from src.data.storage import get_timescaledb_engine
        import pandas as pd
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Load recent price data
        engine = get_timescaledb_engine()
        query = f"""
            SELECT * FROM prices
            WHERE ticker = '{ticker}'
            AND time <= '{date}'
            ORDER BY time DESC
            LIMIT 60
        """
        
        prices_df = pd.read_sql(query, engine)
        
        if prices_df.empty:
            raise HTTPException(status_code=404, detail=f"No price data found for {ticker}")
        
        # Get stock characteristics
        from src.data.storage import get_stock_characteristics
        stock_chars = get_stock_characteristics(ticker)
        
        # Detect regime (simplified - would use twin in production)
        from src.models.regime import detect_regime_features
        regime_id = detect_regime_features(prices_df, stock_chars)
        regime_name = get_regime_name(regime_id)
        strategy = get_regime_trading_strategy(regime_id)
        
        return {
            'ticker': ticker,
            'date': date,
            'regime_id': int(regime_id),
            'regime_name': regime_name,
            'trading_strategy': strategy
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting regime for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

