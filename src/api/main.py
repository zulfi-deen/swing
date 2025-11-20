"""LitServe API for Swing Trading System"""

import litserve as ls
from typing import Dict, Any, Optional
from src.agents.explainer import ExplainerAgent
from src.utils.config import load_config
from src.models.model_registry import ModelRegistry
from src.api.routes import router
import logging

logger = logging.getLogger(__name__)

class SwingTradingAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize models and agents."""
        self.config = load_config()
        self.agent = ExplainerAgent()
        
        # Initialize model registry for side-effects (loading models)
        # In a real distributed setting, this might need to be handled differently
        try:
            self.model_registry = ModelRegistry.get_twin_manager(self.config)
            logger.info("Model registry initialized")
        except Exception as e:
            logger.warning(f"Could not initialize model registry: {e}")

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Decode the incoming request."""
        # Expecting input for 'explain' task
        # { "ticker": "AAPL", "trade_info": {...}, "features": {...} }
        return request

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent to explain the trade."""
        ticker = inputs.get("ticker")
        trade_info = inputs.get("trade_info")
        features = inputs.get("features")
        
        if not ticker or not trade_info:
            # Fallback or error
            return {"error": "Missing ticker or trade_info"}
            
        explanation = self.agent.explain_trade(trade_info, features)
        return {"explanation": explanation, "ticker": ticker}

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the response."""
        return output

if __name__ == "__main__":
    # Create the LitServe API
    api = SwingTradingAPI()
    
    # Create the server
    server = ls.LitServer(api, accelerator="auto", workers_per_device=1)
    
    # Mount the existing FastAPI routes for standard CRUD/DB operations
    # server.app is the underlying FastAPI app
    server.app.include_router(router)
    
    # Run the server
    server.run(port=8000)

