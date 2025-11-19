"""Daily EOD pipeline orchestrator"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import os
import torch

from src.data.ingestion import daily_data_ingestion_flow
from src.data.storage import get_timescaledb_engine, save_to_timescaledb, load_from_local
from src.data.neo4j_client import Neo4jClient
from src.features.technical import compute_all_technical_features
from src.features.cross_sectional import compute_cross_sectional_features, compute_correlation_features
from src.features.graph import build_correlation_graph, build_correlation_graph_from_neo4j
from src.features.options import compute_options_features_batch, get_default_options_features
from src.pipeline.batch_preparation import prepare_batch_for_twin
from src.models.ensemble import ensemble_predictions, LightGBMRanker
from src.models.arima_garch import fit_arima_garch
from src.models.patterns import detect_chart_patterns
from src.models.twin_manager import TwinManager
from src.models.foundation import load_foundation_model
from src.agents.text_summarizer import TextSummarizerAgent
from src.agents.policy_agent import PolicyAgent
from src.agents.explainer import ExplainerAgent
from src.agents.pattern_agent import PatternDetectorAgent
from src.utils.tickers import get_sp500_tickers, get_ticker_sector, get_sector_etf, TICKER_TO_SECTOR
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete daily EOD pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.db_engine = get_timescaledb_engine()
        
        # Initialize Neo4j client
        neo4j_config = self.config.get('storage', {}).get('neo4j', {})
        try:
            self.neo4j_client = Neo4jClient(
                uri=neo4j_config.get('uri'),
                user=neo4j_config.get('user'),
                password=neo4j_config.get('password'),
                database=neo4j_config.get('database')
            )
        except Exception as e:
            logger.warning(f"Could not initialize Neo4j client: {e}")
            self.neo4j_client = None
        
        # Initialize agents
        self.text_agent = TextSummarizerAgent(
            model=self.config['llm_agents']['text_summarizer']['model']
        )
        self.policy_agent = PolicyAgent(
            model=self.config['llm_agents']['policy_agent']['model']
        )
        self.explainer_agent = ExplainerAgent(
            model=self.config['llm_agents']['explainer_agent']['model']
        )
        self.pattern_agent = PatternDetectorAgent()
        
        # Initialize models (would load from disk in production)
        self.lgbm_ranker = self._load_lightgbm_ranker()
        
        # Initialize RL portfolio agent (if enabled)
        self.use_rl_portfolio = self.config.get('rl_portfolio', {}).get('enabled', False)
        self.rl_agent = None
        if self.use_rl_portfolio:
            try:
                from src.agents.portfolio_rl_agent import PortfolioRLAgent
                rl_config = self.config.get('rl_portfolio', {})
                checkpoint_path = rl_config.get('checkpoint_path')
                
                if checkpoint_path and os.path.exists(checkpoint_path):
                    self.rl_agent = PortfolioRLAgent(config=rl_config)
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    self.rl_agent.load_state_dict(checkpoint['agent_state_dict'])
                    self.rl_agent.eval()
                    logger.info(f"Loaded RL portfolio agent from {checkpoint_path}")
                else:
                    logger.warning(f"RL portfolio enabled but checkpoint not found: {checkpoint_path}")
                    self.use_rl_portfolio = False
            except Exception as e:
                logger.error(f"Error loading RL portfolio agent: {e}")
                self.use_rl_portfolio = False
        
        # Initialize TwinManager for digital twins
        try:
            foundation_path = self.config.get('models', {}).get('foundation', {}).get('checkpoint_path')
            pilot_tickers = self.config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
            
            if foundation_path and pilot_tickers:
                foundation_model = load_foundation_model(
                    foundation_path,
                    self.config.get('models', {}).get('foundation', {})
                )
                self.twin_manager = TwinManager(
                    foundation_model=foundation_model,
                    pilot_tickers=pilot_tickers,
                    config=self.config
                )
                logger.info(f"Initialized TwinManager with {len(pilot_tickers)} pilot twins")
            else:
                self.twin_manager = None
                logger.warning("TwinManager not initialized (missing foundation path or pilot tickers)")
        except Exception as e:
            logger.warning(f"Could not initialize TwinManager: {e}")
            self.twin_manager = None
    
    def _load_lightgbm_ranker(self):
        """Load trained LightGBM ranker if it exists."""
        
        import joblib
        from pathlib import Path
        
        model_dir = Path(self.config.get('models', {}).get('lightgbm', {}).get('model_path', 'models/ensemble'))
        model_version = self.config.get('models', {}).get('lightgbm', {}).get('version', 'v1.0')
        model_path = model_dir / f'lgbm_ranker_{model_version}.pkl'
        
        if model_path.exists():
            try:
                ranker = joblib.load(model_path)
                logger.info(f"Loaded LightGBM ranker from {model_path}")
                return ranker
            except Exception as e:
                logger.error(f"Error loading LightGBM ranker: {e}")
                return None
        else:
            logger.warning(f"LightGBM ranker not found at {model_path}. Ensemble will run without it.")
            logger.warning("To train ranker, run: python src/training/train_lightgbm.py")
            return None
        
    def run_daily_pipeline(self, date: Optional[str] = None) -> Dict:
        """
        Run complete daily EOD pipeline.
        
        Args:
            date: Date to process (YYYY-MM-DD). Defaults to today.
        
        Returns:
            Pipeline results dictionary
        """
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Starting daily pipeline for {date}")
        
        tickers = get_sp500_tickers()
        
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        ingestion_result = daily_data_ingestion_flow(date=date, tickers=tickers)
        
        # Step 2: Load data from database
        logger.info("Step 2: Loading data from database")
        prices_df = self._load_prices(date, tickers)
        
        if prices_df.empty:
            logger.error("No price data available. Cannot proceed.")
            return {"error": "No price data"}
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Feature Engineering")
        features_df = self._compute_features(prices_df, date)
        
        # Step 3.1: Detect market regime and add to features
        logger.info("Step 3.1: Detecting market regime")
        spy_data = prices_df[prices_df['ticker'] == 'SPY'] if 'SPY' in prices_df['ticker'].values else pd.DataFrame()
        if not spy_data.empty:
            from src.models.market_regime import get_market_regime_features
            regime_features = get_market_regime_features(spy_data)
            market_regime_id = regime_features['regime']
            regime_name = regime_features['regime_name']
            # Add market regime to all rows
            features_df['market_regime'] = market_regime_id
            features_df['market_regime_name'] = regime_name
            logger.info(f"Market regime detected: {regime_name}")
        else:
            features_df['market_regime'] = 1  # Default: sideways
            features_df['market_regime_name'] = 'Sideways'
            logger.warning("SPY data not available, using default market regime")
        
        # Step 3.5: Build and persist correlation graph to Neo4j
        logger.info("Step 3.5: Building and persisting correlation graph to Neo4j")
        self._build_and_persist_graph(prices_df, date, tickers)
        
        # Step 3.6: Options Features
        logger.info("Step 3.6: Computing options features")
        options_features = self._compute_options_features(tickers, date, prices_df)
        
        # Step 4: Text Features (LLM Agents)
        logger.info("Step 4: Processing text features with LLM agents")
        text_features = self._process_text_features(tickers, date)
        
        # Step 5: Model Inference
        logger.info("Step 5: Model Inference")
        predictions = self._run_model_inference(features_df, text_features, date, options_features)
        
        # Step 6: Ensemble & Ranking
        logger.info("Step 6: Ensemble & Ranking")
        ranked_predictions = self._rank_predictions(predictions, features_df, options_features)
        
        # Step 7: Agent Curation
        logger.info("Step 7: Agent Curation")
        final_recommendations = self._curate_trades(ranked_predictions, date, options_features)
        
        # Step 8: Save to Database
        logger.info("Step 8: Saving recommendations to database")
        self._save_recommendations(final_recommendations, date)
        
        # Step 9: Record paper trades
        logger.info("Step 9: Recording paper trades")
        from src.evaluation.paper_trading import PaperTradingTracker
        paper_tracker = PaperTradingTracker()
        paper_tracker.record_recommendation(date, final_recommendations)
        
        logger.info("Pipeline completed successfully")
        
        return {
            "date": date,
            "tickers_processed": len(tickers),
            "recommendations": len(final_recommendations),
            "ingestion": ingestion_result
        }
    
    def _load_prices(self, date: str, tickers: List[str]) -> pd.DataFrame:
        """Load price data from TimescaleDB."""
        
        # Get last 60 days for feature engineering
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        
        query = """
            SELECT * FROM prices
            WHERE time >= %s AND time <= %s
            AND ticker = ANY(%s)
            ORDER BY time, ticker
        """
        
        try:
            df = pd.read_sql(
                query,
                self.db_engine,
                params=(start_date, date, tickers)
            )
            return df
        except Exception as e:
            logger.error(f"Error loading prices: {e}")
            return pd.DataFrame()
    
    def _compute_features(self, prices_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """Compute all features."""
        
        all_features = []
        
        for ticker in prices_df['ticker'].unique():
            ticker_data = prices_df[prices_df['ticker'] == ticker].copy()
            
            if ticker_data.empty:
                continue
            
            # Technical indicators
            ticker_data = compute_all_technical_features(ticker_data)
            
            all_features.append(ticker_data)
        
        if not all_features:
            return pd.DataFrame()
        
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Cross-sectional features
        features_df = compute_cross_sectional_features(features_df)
        
        # Correlation features
        corr_features = compute_correlation_features(features_df)
        if not corr_features.empty:
            # Ensure date column exists for merge
            if 'date' not in features_df.columns and 'time' in features_df.columns:
                features_df['date'] = pd.to_datetime(features_df['time']).dt.date
            
            merge_cols = ['ticker', 'date'] if 'date' in features_df.columns else ['ticker']
            features_df = features_df.merge(
                corr_features,
                on=merge_cols,
                how='left'
            )
        
        return features_df
    
    def _compute_options_features(
        self,
        tickers: List[str],
        date: str,
        prices_df: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compute options features for all tickers.
        
        Args:
            tickers: List of tickers
            date: Date string
            prices_df: Price DataFrame for context
        
        Returns:
            Dict mapping ticker -> 40 options features
        """
        
        # Check if options are enabled
        options_enabled = self.config.get('data_sources', {}).get('options', {}).get('enabled', False)
        
        if not options_enabled:
            logger.info("Options features disabled in config")
            return {ticker: get_default_options_features() for ticker in tickers}
        
        # Load raw options data from local storage or database
        try:
            options_data_dict = load_from_local(f"raw/options/{date}/options_raw", self.config)
        except:
            logger.warning(f"Could not load raw options data for {date}, using defaults")
            return {ticker: get_default_options_features() for ticker in tickers}
        
        # If we have options data, structure it properly
        structured_options = {}
        for ticker in tickers:
            if ticker in options_data_dict:
                # Options data might be a DataFrame, list of dicts, or dict
                ticker_options = options_data_dict[ticker]
                
                # Convert list of dicts (from JSON) to DataFrame
                if isinstance(ticker_options, list):
                    if len(ticker_options) > 0 and isinstance(ticker_options[0], dict):
                        ticker_options = pd.DataFrame(ticker_options)
                    else:
                        # Empty list - no options data
                        ticker_options = pd.DataFrame()
                
                if isinstance(ticker_options, pd.DataFrame):
                    # Convert DataFrame to structured dict
                    current_price = prices_df[prices_df['ticker'] == ticker]['close'].iloc[-1] if not prices_df[prices_df['ticker'] == ticker].empty else 0.0
                    
                    # Group by strike
                    strikes = []
                    for _, row in ticker_options.iterrows():
                        strike = row.get('strike_price', 0)
                        option_type = row.get('option_type', '').lower()
                        
                        # Find or create strike entry
                        strike_entry = next((s for s in strikes if s.get('strike_price') == strike), None)
                        if not strike_entry:
                            strike_entry = {
                                'strike_price': strike,
                                'expiration': row.get('expiration'),
                                'call_oi': 0, 'put_oi': 0,
                                'call_volume': 0, 'put_volume': 0,
                                'call_delta': 0.0, 'put_delta': 0.0,
                                'call_gamma': 0.0, 'put_gamma': 0.0,
                                'call_vega': 0.0, 'put_vega': 0.0,
                                'call_theta': 0.0, 'put_theta': 0.0,
                                'call_iv': 0.0, 'put_iv': 0.0,
                            }
                            strikes.append(strike_entry)
                        
                        # Fill in data based on option type
                        if option_type == 'call':
                            strike_entry['call_oi'] = int(row.get('open_interest', 0))
                            strike_entry['call_volume'] = int(row.get('volume', 0))
                            strike_entry['call_delta'] = float(row.get('delta', 0.0))
                            strike_entry['call_gamma'] = float(row.get('gamma', 0.0))
                            strike_entry['call_vega'] = float(row.get('vega', 0.0))
                            strike_entry['call_theta'] = float(row.get('theta', 0.0))
                            strike_entry['call_iv'] = float(row.get('implied_volatility', 0.0))
                        elif option_type == 'put':
                            strike_entry['put_oi'] = int(row.get('open_interest', 0))
                            strike_entry['put_volume'] = int(row.get('volume', 0))
                            strike_entry['put_delta'] = float(row.get('delta', 0.0))
                            strike_entry['put_gamma'] = float(row.get('gamma', 0.0))
                            strike_entry['put_vega'] = float(row.get('vega', 0.0))
                            strike_entry['put_theta'] = float(row.get('theta', 0.0))
                            strike_entry['put_iv'] = float(row.get('implied_volatility', 0.0))
                    
                    # Get expirations
                    expirations = sorted(set([s.get('expiration') for s in strikes if s.get('expiration')]))
                    
                    structured_options[ticker] = {
                        'strikes': strikes,
                        'current_price': current_price,
                        'nearest_expiration': expirations[0] if expirations else None,
                        'second_expiration': expirations[1] if len(expirations) > 1 else None,
                        'historical': {}  # Would load from DB in production
                    }
                else:
                    # Already structured
                    structured_options[ticker] = ticker_options
            else:
                # No options data for this ticker
                structured_options[ticker] = {
                    'strikes': [],
                    'current_price': prices_df[prices_df['ticker'] == ticker]['close'].iloc[-1] if not prices_df[prices_df['ticker'] == ticker].empty else 0.0,
                    'historical': {}
                }
        
        # Compute features
        options_features = compute_options_features_batch(
            structured_options,
            prices_df=prices_df,
            date=date
        )
        
        # Save to database
        if options_features:
            options_features_df = pd.DataFrame.from_dict(options_features, orient='index')
            options_features_df['ticker'] = options_features_df.index
            options_features_df['time'] = pd.to_datetime(date)
            options_features_df = options_features_df.reset_index(drop=True)
            
            save_to_timescaledb(options_features_df, table="options_features")
            logger.info(f"Saved options features for {len(options_features)} tickers")
        
        return options_features
    
    def _build_and_persist_graph(
        self,
        prices_df: pd.DataFrame,
        date: str,
        tickers: List[str]
    ):
        """Build correlation graph and persist to Neo4j."""
        
        if not self.neo4j_client:
            logger.warning("Neo4j client not available, skipping graph persistence")
            return
        
        # Build sector map
        sector_map = {ticker: TICKER_TO_SECTOR.get(ticker) for ticker in tickers}
        
        # Build graph and persist to Neo4j
        try:
            build_correlation_graph(
                prices_df,
                date,
                threshold=self.config['models']['gnn'].get('edge_threshold', 0.3),
                persist_to_neo4j=True,
                neo4j_client=self.neo4j_client,
                sector_map=sector_map
            )
            logger.info(f"Successfully persisted correlation graph to Neo4j for {date}")
        except Exception as e:
            logger.error(f"Error building/persisting graph: {e}")
    
    def _process_text_features(
        self,
        tickers: List[str],
        date: str
    ) -> Dict[str, Dict]:
        """Process text features using LLM agents."""
        
        # Load news data from local storage
        try:
            news_data = load_from_local(f"raw/news/{date}/news", self.config)
        except:
            news_data = {}
        
        # Prepare price context
        price_context = {}
        for ticker in tickers:
            # Get recent returns (simplified - would query DB)
            price_context[ticker] = {
                'return_1d': 0.0,
                'return_5d': 0.0,
                'volume_ratio': 1.0,
                'date': date
            }
        
        # Batch process with TextSummarizerAgent
        summaries = self.text_agent.batch_summarize(tickers, news_data, price_context)
        
        # Extract features
        text_features = {}
        for ticker, summary in summaries.items():
            text_features[ticker] = self.text_agent.extract_text_features(summary)
        
        return text_features
    
    def _run_model_inference(
        self,
        features_df: pd.DataFrame,
        text_features: Dict[str, Dict],
        date: str,
        options_features: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Run model inference."""
        
        predictions = []
        
        # Get latest features for each ticker
        if 'date' in features_df.columns:
            latest_features = features_df[features_df['date'] == date]
        elif 'time' in features_df.columns:
            # Convert time to date for filtering
            features_df['date'] = pd.to_datetime(features_df['time']).dt.date
            latest_features = features_df[features_df['date'] == pd.to_datetime(date).date()]
        else:
            latest_features = features_df.groupby('ticker').last().reset_index()
        
        # Load correlation graph from Neo4j (if available)
        graph = None
        ticker_to_idx = {}
        if self.neo4j_client:
            try:
                tickers = latest_features['ticker'].unique().tolist()
                graph, ticker_to_idx = build_correlation_graph_from_neo4j(
                    date=date,
                    tickers=tickers,
                    threshold=self.config['models']['gnn'].get('edge_threshold', 0.3),
                    neo4j_client=self.neo4j_client,
                    lookback_days=30
                )
                logger.info(f"Loaded correlation graph from Neo4j: {len(ticker_to_idx)} nodes")
            except Exception as e:
                logger.warning(f"Could not load graph from Neo4j: {e}. Computing from scratch.")
                graph = None
        
        # Use TwinManager for pilot stocks, fallback to LightGBM for others
        pilot_tickers = self.config.get('models', {}).get('twins', {}).get('pilot_tickers', [])
        
        for _, row in latest_features.iterrows():
            ticker = row['ticker']
            
            # ARIMA/GARCH baseline
            returns = features_df[features_df['ticker'] == ticker]['close'].pct_change().dropna()
            arima_garch = fit_arima_garch(ticker, returns) if len(returns) > 50 else {}
            
            # Pattern detection
            ticker_data = features_df[features_df['ticker'] == ticker].tail(60)
            patterns = detect_chart_patterns(ticker_data)
            
            # LightGBM ranking
            lgbm_rank = None
            if self.lgbm_ranker is not None:
                try:
                    # Prepare features for LightGBM
                    lgbm_features = row[[
                        'rsi_14', 'macd', 'macd_signal', 'atr_14',
                        'volume_z_score', 'return_rank_5d', 'return_rank_20d',
                        'sector_relative_strength', 'correlation_to_spy',
                        'sentiment_score', 'pattern_confidence'
                    ]].fillna(0).values.reshape(1, -1)
                    
                    lgbm_rank = self.lgbm_ranker.model.predict(lgbm_features)[0]
                except Exception as e:
                    logger.warning(f"Error getting LightGBM rank for {ticker}: {e}")
                    lgbm_rank = None
            
            # Use TwinManager if available and ticker is in pilot list
            if self.twin_manager and ticker in pilot_tickers:
                try:
                    twin = self.twin_manager.get_twin(ticker)
                    if twin:
                        # Prepare batch for twin using proper batch preparation
                        batch = prepare_batch_for_twin(
                            features_df,
                            ticker,
                            normalizer=None,  # Could add normalizer if needed
                            config=self.config
                        )
                        
                        # Get options features tensor if available
                        options_tensor = None
                        if options_features and ticker in options_features:
                            import torch
                            # Convert options features dict to tensor (40 features)
                            opts = options_features[ticker]
                            options_list = [
                                opts.get('call_oi', 0), opts.get('put_oi', 0), opts.get('total_oi', 0),
                                opts.get('call_volume', 0), opts.get('put_volume', 0), opts.get('total_volume', 0),
                                opts.get('oi_change_pct', 0.0), opts.get('volume_zscore', 0.0),
                                opts.get('pcr_oi', 1.0), opts.get('pcr_volume', 1.0), opts.get('pcr_zscore', 0.0),
                                float(opts.get('pcr_extreme_bullish', False)), float(opts.get('pcr_extreme_bearish', False)),
                                opts.get('pcr_change', 0.0),
                                opts.get('max_pain_strike', 0.0), opts.get('max_pain_distance_pct', 0.0),
                                opts.get('total_gamma', 0.0), float(opts.get('gamma_sign', 0)),
                                opts.get('gamma_concentration', 0.0),
                                opts.get('gamma_flip_strike', 0.0) if opts.get('gamma_flip_strike') else 0.0,
                                opts.get('gamma_flip_distance_pct', 0.0) if opts.get('gamma_flip_distance_pct') else 0.0,
                                opts.get('atm_call_iv', 0.25), opts.get('atm_put_iv', 0.25),
                                opts.get('iv_skew', 0.0), opts.get('put_call_iv_ratio', 1.0),
                                opts.get('iv_percentile', 0.5), opts.get('iv_rank', 0.5), opts.get('iv_change_pct', 0.0),
                                opts.get('net_delta', 0.0), opts.get('net_gamma', 0.0), opts.get('net_vega', 0.0),
                                opts.get('net_theta', 0.0), opts.get('net_delta_abs', 0.0),
                                opts.get('front_month_oi', 0), opts.get('next_month_oi', 0),
                                opts.get('roll_ratio', 1.0), opts.get('term_curve_slope', 0.0),
                                opts.get('trend_signal', 0.0), opts.get('sentiment_signal', 0.0),
                                float(opts.get('gamma_signal', 0))
                            ]
                            options_tensor = torch.tensor(options_list, dtype=torch.float32).unsqueeze(0)  # (1, 40)
                        
                        # Get twin predictions
                        twin_preds = twin.forward(batch, graph, options_features=options_tensor)
                        
                        tft_gnn_preds = {
                            'return': float(twin_preds['expected_return'].item() if hasattr(twin_preds['expected_return'], 'item') else twin_preds['expected_return']),
                            'prob_hit_long': float(twin_preds['hit_prob'].item() if hasattr(twin_preds['hit_prob'], 'item') else twin_preds['hit_prob']),
                            'volatility': float(twin_preds['volatility'].item() if hasattr(twin_preds['volatility'], 'item') else twin_preds['volatility']),
                            'quantiles': {
                                'q10': float(twin_preds['quantiles']['q10'].item() if hasattr(twin_preds['quantiles']['q10'], 'item') else twin_preds['quantiles']['q10']),
                                'q50': float(twin_preds['quantiles']['q50'].item() if hasattr(twin_preds['quantiles']['q50'], 'item') else twin_preds['quantiles']['q50']),
                                'q90': float(twin_preds['quantiles']['q90'].item() if hasattr(twin_preds['quantiles']['q90'], 'item') else twin_preds['quantiles']['q90'])
                            },
                            'regime': int(twin_preds['regime'].item() if hasattr(twin_preds['regime'], 'item') else twin_preds['regime'])
                        }
                    else:
                        # Fallback if twin not available
                        tft_gnn_preds = {
                            'return': 0.0,
                            'prob_hit_long': 0.5,
                            'volatility': 0.02,
                            'quantiles': {'q10': -0.05, 'q50': 0.0, 'q90': 0.05}
                        }
                except Exception as e:
                    logger.warning(f"Error getting twin prediction for {ticker}: {e}, using fallback")
                    tft_gnn_preds = {
                        'return': 0.0,
                        'prob_hit_long': 0.5,
                        'volatility': 0.02,
                        'quantiles': {'q10': -0.05, 'q50': 0.0, 'q90': 0.05}
                    }
            else:
                # Fallback for non-pilot stocks or if TwinManager not available
                tft_gnn_preds = {
                    'return': 0.0,
                    'prob_hit_long': 0.5,
                    'volatility': 0.02,
                    'quantiles': {'q10': -0.05, 'q50': 0.0, 'q90': 0.05}
                }
            
            # Ensemble
            ensemble_pred = ensemble_predictions(
                tft_gnn_preds,
                lgbm_ranks=lgbm_rank,
                arima_garch=arima_garch,
                patterns=patterns,
                text_features=text_features.get(ticker, {})
            )
            
            predictions.append({
                'ticker': ticker,
                'date': date,
                'expected_return': ensemble_pred['expected_return'],
                'hit_probability': ensemble_pred['hit_probability'],
                'volatility': ensemble_pred['volatility'],
                'sector': get_ticker_sector(ticker),
                'regime': tft_gnn_preds.get('regime', None),
                **text_features.get(ticker, {})
            })
        
        return pd.DataFrame(predictions)
    
    def _rank_predictions(
        self,
        predictions: pd.DataFrame,
        features_df: pd.DataFrame,
        options_features: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Rank predictions using RL agent or fallback to priority scoring."""
        
        if predictions.empty:
            return predictions
        
        # Use RL agent if enabled and available
        if self.use_rl_portfolio and self.rl_agent is not None:
            try:
                return self._rank_with_rl_agent(predictions, features_df, options_features)
            except Exception as e:
                logger.error(f"Error in RL ranking, falling back to priority scoring: {e}")
        
        # Fallback: Calculate priority score (legacy method)
        predictions['priority_score'] = (
            predictions['expected_return'] *
            predictions['hit_probability'] *
            (1 - predictions['volatility'] * 10)  # Scale volatility
        )
        
        # Sort by priority
        predictions = predictions.sort_values('priority_score', ascending=False)
        
        return predictions
    
    def _rank_with_rl_agent(
        self,
        predictions: pd.DataFrame,
        features_df: pd.DataFrame,
        options_features: Optional[Dict[str, Dict]] = None
    ) -> pd.DataFrame:
        """Rank predictions using RL portfolio agent."""
        
        from src.data.rl_state_builder import RLStateBuilder
        import torch
        
        # Build RL state
        state_builder = RLStateBuilder(self.config)
        
        # Convert predictions to twin_predictions format
        twin_predictions = {}
        for _, row in predictions.iterrows():
            ticker = row['ticker']
            twin_predictions[ticker] = {
                'expected_return': float(row.get('expected_return', 0.0)),
                'hit_prob': float(row.get('hit_probability', 0.5)),
                'volatility': float(row.get('volatility', 0.02)),
                'regime': int(row.get('regime', 0)),
                'idiosyncratic_alpha': 0.0,
                'quantile_10': float(row.get('quantile_10', -0.05)),
                'quantile_90': float(row.get('quantile_90', 0.05)),
            }
        
        # Get portfolio state (simplified - would come from actual portfolio tracker)
        portfolio_state = {
            'cash': 0.35,
            'num_positions': 0,
            'positions': {},
            'sector_exposure': {},
            'portfolio_value': 100000.0,
            'peak_value': 100000.0,
        }
        
        # Get macro context (simplified - would come from actual macro data)
        macro_context = {
            'vix': 20.0,
            'spy_return_5d': 0.0,
            'treasury_10y': 4.0,
            'market_regime': 'bull',
        }
        
        # Build state (include options features if available)
        state = state_builder.build_state(
            twin_predictions=twin_predictions,
            features_df=features_df,
            portfolio_state=portfolio_state,
            macro_context=macro_context,
            prices_df=None,  # Would use actual prices in production
            date=None,
            neo4j_client=self.neo4j_client,
            options_features=options_features
        )
        
        # Get action from RL agent
        with torch.no_grad():
            action, _, _, _ = self.rl_agent(state, deterministic=True)
        
        # Extract attention scores and position sizes
        stock_selection = action.get('stock_selection', {})
        position_sizes = action.get('position_sizes', {})
        selected_indices = action.get('selected_indices', [])
        tickers = state.get('tickers', [])
        
        # Fallback: use predictions tickers if state tickers don't match
        if not tickers or len(tickers) != len(predictions):
            tickers = predictions['ticker'].tolist()
        
        # Map to predictions DataFrame
        if isinstance(stock_selection, torch.Tensor):
            # Convert tensor to dict
            if len(tickers) == len(stock_selection):
                attention_scores = {tickers[i]: float(stock_selection[i]) for i in range(len(tickers))}
            else:
                # Mismatch: map by index or use default
                attention_scores = {ticker: 0.0 for ticker in tickers}
                for i in range(min(len(tickers), len(stock_selection))):
                    attention_scores[tickers[i]] = float(stock_selection[i])
        else:
            attention_scores = stock_selection if isinstance(stock_selection, dict) else {}
        
        if isinstance(position_sizes, torch.Tensor):
            if len(tickers) == len(position_sizes):
                position_sizes_dict = {tickers[i]: float(position_sizes[i]) for i in range(len(tickers))}
            else:
                position_sizes_dict = {ticker: 0.0 for ticker in tickers}
                for i in range(min(len(tickers), len(position_sizes))):
                    position_sizes_dict[tickers[i]] = float(position_sizes[i])
        else:
            position_sizes_dict = position_sizes if isinstance(position_sizes, dict) else {}
        
        # Add RL scores to predictions
        predictions['rl_attention_score'] = predictions['ticker'].map(attention_scores).fillna(0.0)
        predictions['rl_position_size'] = predictions['ticker'].map(position_sizes_dict).fillna(0.0)
        
        # Use RL attention score as priority score
        predictions['priority_score'] = predictions['rl_attention_score']
        
        # Sort by RL attention score
        predictions = predictions.sort_values('priority_score', ascending=False)
        
        return predictions
    
    def _curate_trades(
        self,
        predictions: pd.DataFrame,
        date: str,
        options_features: Optional[Dict[str, Dict]] = None
    ) -> List[Dict]:
        """Curate final trades using PolicyAgent."""
        
        if predictions.empty:
            return []
        
        # Prepare candidates DataFrame
        candidates = predictions.copy()
        candidates['side'] = candidates['expected_return'].apply(
            lambda x: 'buy' if x > 0 else 'sell'
        )
        candidates['target_pct'] = candidates['expected_return'].abs()
        candidates['stop_pct'] = candidates['volatility'] * 1.5  # Stop at 1.5x volatility
        
        # Portfolio state (simplified)
        portfolio_state = {
            'positions': [],
            'sector_exposure': {},
            'cash': 100000
        }
        
        # Risk rules from config
        risk_rules = self.config['trading_rules']
        
        # Curate trades
        trades = self.policy_agent.curate_trades(
            candidates,
            portfolio_state,
            risk_rules,
            date,
            max_trades=risk_rules.get('max_positions', 15)
        )
        
        # Add explanations (with options if available)
        for trade in trades:
            try:
                ticker = trade['ticker']
                ticker_features = predictions[predictions['ticker'] == ticker].iloc[0].to_dict()
                
                # Use options-aware explanation if options features available
                if options_features and ticker in options_features:
                    # Get twin predictions (would come from actual twin in production)
                    twin_pred = {
                        'expected_return': ticker_features.get('expected_return', 0.0),
                        'hit_prob': ticker_features.get('hit_probability', 0.5),
                        'volatility': ticker_features.get('volatility', 0.02),
                    }
                    explanation = self.explainer_agent.explain_trade_with_options(
                        trade, twin_pred, options_features[ticker]
                    )
                else:
                    explanation = self.explainer_agent.explain_trade(trade, ticker_features)
                
                trade['explanation'] = explanation
            except Exception as e:
                logger.error(f"Error generating explanation for {trade['ticker']}: {e}")
                trade['explanation'] = "Explanation unavailable"
        
        return trades
    
    def _save_recommendations(self, recommendations: List[Dict], date: str):
        """Save recommendations to database."""
        
        if not recommendations:
            return
        
        records = []
        for trade in recommendations:
            records.append({
                'date': date,
                'ticker': trade['ticker'],
                'side': trade['side'],
                'target_pct': trade['target_pct'],
                'stop_pct': trade['stop_pct'],
                'probability': trade['probability'],
                'priority_score': trade.get('priority_score', 0.0),
                'position_size_pct': trade['position_size_pct'],
                'rationale': trade.get('rationale', []),
                'model_version': 'v1.0'
            })
        
        df = pd.DataFrame(records)
        save_to_timescaledb(df, table="recommendations", if_exists="replace")


def run_daily_pipeline(date: Optional[str] = None) -> Dict:
    """Convenience function to run the pipeline."""
    
    orchestrator = PipelineOrchestrator()
    return orchestrator.run_daily_pipeline(date)
