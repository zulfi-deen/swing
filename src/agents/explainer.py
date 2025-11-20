"""ExplainerAgent - Generate human-readable rationale"""

import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)


class ExplainerAgent:
    """LLM agent to generate explanations for trade recommendations."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a trading analyst explaining trade recommendations.

Your job: Generate clear, concise rationale for why a trade is recommended.

Focus on:
- Key technical signals (breakouts, patterns, indicators)
- Fundamental catalysts (earnings, news, events)
- Risk factors and stop levels
- Expected price movement and timeline

Be specific. Avoid jargon. Write for a retail trader."""),
            ("user", """Ticker: {ticker}
Side: {side}
Target: {target_pct:.1%}
Stop: {stop_pct:.1%}
Probability: {probability:.1%}

Features:
{features}

Generate 3-5 bullet points explaining this recommendation.""")
        ])
    
    def explain_trade(self, trade: Dict, features: Dict) -> str:
        """Generate explanation for a trade."""
        
        messages = self.prompt.format_messages(
            ticker=trade.get('ticker', ''),
            side=trade.get('side', 'buy'),
            target_pct=trade.get('target_pct', 0.05),
            stop_pct=trade.get('stop_pct', 0.03),
            probability=trade.get('probability', 0.6),
            features=self._format_features(features)
        )
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Explanation unavailable."
    
    def _format_features(self, features: Dict) -> str:
        """Format features dict as text."""
        
        lines = []
        for key, value in features.items():
            if isinstance(value, (int, float)):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def explain_trade_with_options(
        self,
        trade: Dict,
        twin_pred: Dict,
        options_feat: Dict
    ) -> str:
        """
        Generate explanation for a trade including options context.
        
        Args:
            trade: Trade dict with ticker, side, target, stop, etc.
            twin_pred: Twin prediction dict
            options_feat: Options features dict (40 features)
        
        Returns:
            Explanation string
        """
        
        options_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a trading analyst explaining trade recommendations with options market intelligence.

Your job: Generate clear, concise rationale incorporating options signals.

Focus on:
- Twin predictions (expected return, probability, regime)
- Options signals (PCR extremes, gamma zones, IV regime, OI trends)
- How options confirm or contradict the equity signal
- Risk factors from options (max pain, IV spikes, sentiment extremes)

Be specific. Explain options concepts simply. Write for a retail trader."""),
            ("user", """Ticker: {ticker}
Side: {side}
Target: {target_pct:.1%}
Stop: {stop_pct:.1%}
Probability: {probability:.1%}
Regime: {regime}

Twin Predictions:
- Expected Return: {expected_return:.2%}
- Hit Probability: {hit_prob:.1%}
- Volatility: {volatility:.2%}

Options Context:
- PCR: {pcr_oi:.2f} (z-score: {pcr_zscore:.2f})
- PCR Extreme: {pcr_extreme}
- Max Pain: ${max_pain_strike:.2f} ({max_pain_distance_pct:.1%} away)
- Gamma Signal: {gamma_signal}
- IV Percentile: {iv_percentile:.0%}
- Net Delta: {net_delta:.3f}
- Trend Signal: {trend_signal:.1f}
- Sentiment Signal: {sentiment_signal:.1f}

Generate 4-6 bullet points explaining this recommendation with options context.""")
        ])
        
        pcr_extreme = "None"
        if options_feat.get('pcr_extreme_bullish'):
            pcr_extreme = "Extreme Bullish (contrarian bearish)"
        elif options_feat.get('pcr_extreme_bearish'):
            pcr_extreme = "Extreme Bearish (contrarian bullish)"
        
        messages = options_prompt.format_messages(
            ticker=trade.get('ticker', ''),
            side=trade.get('side', 'buy'),
            target_pct=trade.get('target_pct', 0.05),
            stop_pct=trade.get('stop_pct', 0.03),
            probability=trade.get('probability', 0.6),
            regime=trade.get('regime', 'Unknown'),
            expected_return=twin_pred.get('expected_return', 0.0),
            hit_prob=twin_pred.get('hit_prob', 0.5),
            volatility=twin_pred.get('volatility', 0.02),
            pcr_oi=options_feat.get('pcr_oi', 1.0),
            pcr_zscore=options_feat.get('pcr_zscore', 0.0),
            pcr_extreme=pcr_extreme,
            max_pain_strike=options_feat.get('max_pain_strike', 0.0),
            max_pain_distance_pct=options_feat.get('max_pain_distance_pct', 0.0),
            gamma_signal="Active" if options_feat.get('gamma_signal') else "Inactive",
            iv_percentile=options_feat.get('iv_percentile', 0.5),
            net_delta=options_feat.get('net_delta', 0.0),
            trend_signal=options_feat.get('trend_signal', 0.0),
            sentiment_signal=options_feat.get('sentiment_signal', 0.0),
        )
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error generating options explanation: {e}")
            return "Options explanation unavailable."

