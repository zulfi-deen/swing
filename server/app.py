import lightning as L
import litserve as ls
import os
import logging

# Import API components
from src.api.main import SwingTradingAPI
from src.api.routes import router

# Import Training components
from src.training.train_foundation import train_foundation_model
from src.training.train_twins_lightning import fine_tune_twin
from src.models.foundation import load_foundation_model
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class SwingAPIService(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=True)

    def run(self):
        # Initialize LitServe API
        api = SwingTradingAPI()
        
        # Create server
        # accelerator="auto" will use GPU if available
        server = ls.LitServer(api, accelerator="auto", workers_per_device=1)
        
        # Attach the existing FastAPI routes (legacy endpoints)
        server.app.include_router(router)
        
        # Run server on the port assigned by Lightning
        logger.info(f"Starting Swing API on port {self.port}")
        server.run(port=self.port)

class TrainingWork(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=True, cache_calls=True)
        
    def train_foundation(self, use_synthetic=False):
        """Run foundation model training."""
        logger.info("Starting foundation model training job...")
        config = load_config()
        train_foundation_model(
            config_path=None, # Load default
            use_synthetic_data=use_synthetic
        )
        logger.info("Foundation model training completed.")

    def train_twin(self, ticker: str, lookback_days: int = 180):
        """Run digital twin fine-tuning."""
        logger.info(f"Starting twin training for {ticker}...")
        config = load_config()
        
        foundation_path = config.get('models', {}).get('foundation', {}).get('checkpoint_path')
        if not foundation_path or not os.path.exists(foundation_path):
            logger.error(f"Foundation checkpoint not found at {foundation_path}")
            return
            
        foundation_model = load_foundation_model(foundation_path, config.get('models', {}).get('foundation', {}))
        
        fine_tune_twin(
            ticker=ticker,
            foundation_model=foundation_model,
            config=config,
            lookback_days=lookback_days
        )
        logger.info(f"Twin training for {ticker} completed.")

class SwingTradingApp(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.api_service = SwingAPIService()
        self.training_service = TrainingWork()
        
    def run(self):
        # Run the API service
        self.api_service.run()
        
        # Training is triggered manually or via API interaction (if implemented)
        # For now, it just sits ready to accept calls via the UI/CLI if we expose them
        
    def configure_layout(self):
        return L.frontend.StreamlitFrontend(render_fn=render_fn)

def render_fn(state):
    import streamlit as st
    
    st.title("Swing Trading Control Plane")
    
    st.header("API Status")
    if state.api_service.url:
        st.success(f"API running at {state.api_service.url}")
    else:
        st.warning("API starting...")
        
    st.header("Training Jobs")
    
    st.subheader("Foundation Model")
    if st.button("Train Foundation Model"):
        state.training_service.train_foundation(use_synthetic=True)
        st.info("Job triggered")
        
    st.subheader("Digital Twins")
    ticker = st.text_input("Ticker Symbol", "AAPL")
    if st.button("Train Twin"):
        state.training_service.train_twin(ticker=ticker)
        st.info(f"Job triggered for {ticker}")

app = L.LightningApp(SwingTradingApp())

