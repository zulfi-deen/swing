"""PatternDetectorAgent - Analyze chart patterns"""

import pandas as pd
from typing import Dict, List
from src.models.patterns import detect_chart_patterns
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import json
import os

class PatternDetectorAgent:
    """
    Agent that combines rule-based pattern detection with LLM analysis.
    It validates technical patterns found by the quantitative model.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key) if api_key else None
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a technical analyst verifying chart patterns.
            
Input: A technical pattern detected by algorithms (e.g., "Double Bottom") and recent price context.
Task: Assess the quality of the pattern. Is it clean? Is volume confirming it?
Output: JSON with 'confirmed' (bool), 'confidence' (0-1), and 'analysis' (string)."""),
            ("user", """Pattern Detected: {pattern_type}
Confidence: {confidence}
Context:
- Price change 5d: {return_5d:.2%}
- Volume trend: {volume_trend}
- Distance to resistance: {dist_resistance:.2%}

Verify this pattern.""")
        ])

    def analyze(self, ticker: str, df: pd.DataFrame) -> Dict:
        # 1. Run rule-based detection
        patterns = detect_chart_patterns(df)
        
        detected_pattern = "None"
        confidence = 0.0
        
        if patterns.get('double_bottom'):
            detected_pattern = "Double Bottom"
            confidence = patterns['confidence']
        elif patterns.get('breakout_52w_high'):
            detected_pattern = "52w Breakout"
            confidence = patterns['confidence']
            
        if detected_pattern == "None" or not self.llm:
            return {
                "ticker": ticker,
                "pattern": "None",
                "confirmed": False,
                "confidence": 0.0,
                "analysis": "No significant patterns detected."
            }

        # 2. LLM Verification (Optional enhancement)
        # Calculate context metrics for the LLM
        return_5d = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        vol_curr = df['volume'].iloc[-1]
        volume_trend = "High" if vol_curr > vol_mean * 1.2 else "Normal"
        
        try:
            messages = self.prompt.format_messages(
                pattern_type=detected_pattern,
                confidence=confidence,
                return_5d=return_5d,
                volume_trend=volume_trend,
                dist_resistance=0.05 # Placeholder
            )
            response = self.llm.invoke(messages)
            analysis = json.loads(response.content)
            return {
                "ticker": ticker,
                "pattern": detected_pattern,
                "confirmed": analysis.get("confirmed", False),
                "confidence": analysis.get("confidence", confidence),
                "analysis": analysis.get("analysis", "")
            }
        except:
            return {
                "ticker": ticker,
                "pattern": detected_pattern,
                "confirmed": True, # Fallback to algorithmic confirmation
                "confidence": confidence,
                "analysis": "Algorithmic detection only."
            }

