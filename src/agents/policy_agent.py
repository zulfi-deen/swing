"""PolicyAgent - Apply trading rules and curate final trades"""

import json
import os
import pandas as pd
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import logging

logger = logging.getLogger(__name__)


class PolicyAgent:
    """LLM agent to apply trading policy and curate final trades."""
    
    def __init__(self, model: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.llm = ChatOpenAI(model=model, temperature=0.2, api_key=api_key)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a risk-aware swing trading strategist.
Your job: Review candidate trades from quantitative models and select the best 10-15 trades.

Criteria:
1. Have high probability (>60%)
2. Show technical confirmation (breakouts, patterns, indicators aligned)
3. Avoid negative divergences (model says buy but news is terrible)
4. Maintain portfolio diversification (sectors, factors)
5. Respect risk limits (max positions, sector exposure, position sizing)

Output: JSON array of selected trades with rationale for each.
Be selective. Quality > quantity."""),
            ("user", """Date: {date}
Top Candidates:
{candidates_table}
Current Portfolio:
- Open positions: {num_positions}
- Sector exposure: {sector_exposure}
- Available capital: ${cash:,.0f}

Risk Rules:
- Max 15 concurrent positions
- Max 25% allocation per sector
- Min probability: 60%
- Avoid: earnings in next 2 days, extreme volatility

Select up to {max_trades} best trades. For each trade, output:
{{
  "ticker": "...",
  "side": "buy" or "sell",
  "target_pct": <float>,
  "stop_pct": <float>,
  "probability": <float>,
  "position_size_pct": <suggested % of portfolio>,
  "rationale": ["reason 1", "reason 2", "reason 3"]
}}
Output JSON array.""")
        ])
    
    def curate_trades(
        self,
        candidates: pd.DataFrame,
        portfolio_state: Dict,
        risk_rules: Dict,
        date: str,
        max_trades: int = 12
    ) -> List[Dict]:
        """Curate final trade list from candidates."""
        
        # First, apply hard filters
        candidates = self._apply_hard_filters(candidates, risk_rules)
        
        if candidates.empty:
            return []
        
        # Format candidates table for LLM
        candidates_table = self._format_candidates_table(candidates)
        
        # Call LLM
        messages = self.prompt.format_messages(
            date=date,
            candidates_table=candidates_table,
            num_positions=len(portfolio_state.get('positions', [])),
            sector_exposure=json.dumps(portfolio_state.get('sector_exposure', {}), indent=2),
            cash=portfolio_state.get('cash', 100000),
            max_trades=max_trades
        )
        
        try:
            response = self.llm.invoke(messages)
            
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            trades = json.loads(content)
        except Exception as e:
            logger.error(f"Error parsing PolicyAgent response: {e}")
            # Fallback: top trades by priority score
            trades = self._fallback_selection(candidates, max_trades)
        
        return trades
    
    def _apply_hard_filters(self, candidates: pd.DataFrame, rules: Dict) -> pd.DataFrame:
        """Apply non-negotiable filters."""
        
        filtered = candidates.copy()
        
        if 'avg_dollar_volume' in filtered.columns:
            filtered = filtered[filtered['avg_dollar_volume'] > rules.get('min_dollar_volume', 5_000_000)]
        
        if 'bid_ask_spread_pct' in filtered.columns:
            filtered = filtered[filtered['bid_ask_spread_pct'] < rules.get('max_bid_ask_spread', 0.005)]
        
        if 'probability' in filtered.columns:
            filtered = filtered[filtered['probability'] >= rules.get('min_probability', 0.60)]
        
        if 'days_to_earnings' in filtered.columns:
            filtered = filtered[filtered['days_to_earnings'] > rules.get('days_to_earnings_avoid', 2)]
        
        if 'volatility' in filtered.columns:
            filtered = filtered[filtered['volatility'] < rules.get('max_volatility', 0.05)]
        
        return filtered
    
    def _format_candidates_table(self, df: pd.DataFrame) -> str:
        """Format DataFrame as text table for LLM."""
        
        cols = [
            'ticker', 'side', 'expected_return', 'probability',
            'priority_score', 'sector', 'sentiment_score',
        ]
        
        available_cols = [c for c in cols if c in df.columns]
        table = df[available_cols].head(50).to_string(index=False, max_colwidth=30)
        
        return table
    
    def _fallback_selection(self, candidates: pd.DataFrame, max_trades: int) -> List[Dict]:
        """Fallback selection by priority score."""
        
        if 'priority_score' in candidates.columns:
            top = candidates.nlargest(max_trades, 'priority_score')
        else:
            top = candidates.head(max_trades)
        
        trades = []
        for _, row in top.iterrows():
            trades.append({
                'ticker': row.get('ticker', ''),
                'side': row.get('side', 'buy'),
                'target_pct': float(row.get('target_pct', 0.05)),
                'stop_pct': float(row.get('stop_pct', 0.03)),
                'probability': float(row.get('probability', 0.6)),
                'position_size_pct': float(row.get('position_size_pct', 0.05)),
                'rationale': ['Selected by priority score']
            })
        
        return trades

