"""TextSummarizerAgent - Process news/text into structured features"""

import json
import os
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TextSummary(BaseModel):
    """Structured output from TextSummarizerAgent."""
    
    ticker: str
    sentiment_score: float = Field(description="Sentiment from -1 (bearish) to 1 (bullish)")
    sentiment_confidence: float = Field(description="Confidence 0-1")
    key_narratives: List[str] = Field(description="2-3 key themes")
    event_flags: Dict = Field(description="Binary flags for key events")
    news_intensity: str = Field(description="quiet | moderate | high")
    contrarian_signals: List[str] = Field(description="Text-price divergences")


class TextSummarizerAgent:
    """LLM agent to process news/text into structured features."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.llm = ChatOpenAI(model=model, temperature=0.1, api_key=api_key)
        
        # Initialize sentence transformer for narrative embeddings
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dim embeddings
            logger.info(f"Loaded sentence transformer with {self.embedding_dim}-dim embeddings")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.embedder = None
            self.embedding_dim = 384
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial text analyzer specializing in swing trading.

Your job: Analyze news headlines, analyst changes, and social sentiment for a stock, then output structured JSON.

Key focus areas:
- Sentiment (bullish/bearish/neutral)
- Event-driven catalysts (earnings, guidance, management changes, regulatory)
- Narrative themes (e.g., "AI demand", "margin pressure", "sector rotation")
- Contrarian signals (stock price contradicts news sentiment)

Output format: JSON matching TextSummary schema.
Be concise. Avoid generic statements. Focus on actionable signals."""),
            ("user", """Ticker: {ticker}
Date: {date}
Headlines (last 24h):
{headlines}
Analyst Changes:
{analyst_changes}
Recent Price Action:
- 1-day return: {return_1d:.2%}
- 5-day return: {return_5d:.2%}
- Volume vs avg: {volume_ratio:.1f}x

Analyze and output JSON.""")
        ])
    
    def summarize(
        self,
        ticker: str,
        headlines: List[str],
        analyst_changes: List[Dict],
        price_context: Dict,
        date: str
    ) -> TextSummary:
        """Summarize text data for a ticker."""
        
        # Format inputs
        headlines_str = "\n".join([f"- {h}" for h in headlines]) if headlines else "No news"
        
        analyst_str = "\n".join([
            f"- {a.get('firm', 'Unknown')}: {a.get('action', '')} {a.get('rating', '')} "
            f"(target ${a.get('target', 'N/A')})"
            for a in analyst_changes
        ]) if analyst_changes else "No changes"
        
        # Call LLM
        messages = self.prompt.format_messages(
            ticker=ticker,
            date=date,
            headlines=headlines_str,
            analyst_changes=analyst_str,
            return_1d=price_context.get('return_1d', 0.0),
            return_5d=price_context.get('return_5d', 0.0),
            volume_ratio=price_context.get('volume_ratio', 1.0)
        )
        
        try:
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            data = json.loads(content)
            summary = TextSummary(**data)
        except Exception as e:
            logger.error(f"Error parsing LLM response for {ticker}: {e}")
            # Fallback to neutral
            summary = TextSummary(
                ticker=ticker,
                sentiment_score=0.0,
                sentiment_confidence=0.5,
                key_narratives=["No significant news"],
                event_flags={},
                news_intensity="quiet",
                contrarian_signals=[]
            )
        
        return summary
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate semantic embeddings for narrative text using sentence-transformers.
        
        Args:
            text: Narrative text to embed
        
        Returns:
            numpy array with embedding (384-dim for all-MiniLM-L6-v2)
        """
        if not text or len(text.strip()) == 0:
            return np.zeros(self.embedding_dim)
        
        if self.embedder is None:
            logger.warning("Sentence transformer not available, returning zero vector")
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
            # Ensure correct dimension
            if len(embedding) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                if len(embedding) > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                else:
                    padding = np.zeros(self.embedding_dim - len(embedding))
                    embedding = np.concatenate([embedding, padding])
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return np.zeros(self.embedding_dim)
    
    def extract_text_features(self, summary: TextSummary) -> Dict:
        """
        Convert LLM summary to model features including embeddings.
        
        Args:
            summary: TextSummary object from LLM
        
        Returns:
            Dictionary of features including narrative embeddings
        """
        features = {
            'sentiment_score': summary.sentiment_score,
            'sentiment_confidence': summary.sentiment_confidence,
            'news_intensity_quiet': summary.news_intensity == 'quiet',
            'news_intensity_moderate': summary.news_intensity == 'moderate',
            'news_intensity_high': summary.news_intensity == 'high',
        }
        
        # Event flags
        event_flags = summary.event_flags or {}
        features['event_flag_earnings'] = event_flags.get('earnings_surprise', False)
        features['event_flag_guidance'] = event_flags.get('guidance_change', False)
        features['event_flag_regulatory'] = event_flags.get('regulatory_risk', False)
        features['has_contrarian_signal'] = len(summary.contrarian_signals) > 0
        
        # Narrative embeddings (384-dim for all-MiniLM-L6-v2)
        narrative_text = " ".join(summary.key_narratives)
        narrative_embedding = self.embed_text(narrative_text)
        
        # Store embedding values (first 32 for backward compatibility, or all if needed)
        # In production, might want to use PCA or store full embedding separately
        for i, val in enumerate(narrative_embedding[:32]):  # Store first 32 for now
            features[f'narrative_embed_{i}'] = float(val)
        
        # Store full embedding as a separate field for future use
        features['narrative_embedding_full'] = narrative_embedding.tolist()
        
        return features
    
    def batch_summarize(
        self,
        tickers: List[str],
        news_data: Dict[str, Dict],
        price_data: Dict[str, Dict]
    ) -> Dict[str, TextSummary]:
        """Process multiple tickers in parallel."""
        
        from concurrent.futures import ThreadPoolExecutor
        
        def process_one(ticker: str) -> tuple:
            try:
                summary = self.summarize(
                    ticker=ticker,
                    headlines=news_data.get(ticker, {}).get('headlines', []),
                    analyst_changes=news_data.get(ticker, {}).get('analyst_changes', []),
                    price_context=price_data.get(ticker, {}),
                    date=price_data.get(ticker, {}).get('date', '')
                )
                return ticker, summary
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                # Return neutral summary on error
                return ticker, TextSummary(
                    ticker=ticker,
                    sentiment_score=0.0,
                    sentiment_confidence=0.5,
                    key_narratives=["No significant news"],
                    event_flags={},
                    news_intensity="quiet",
                    contrarian_signals=[]
                )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_one, tickers))
        
        return {ticker: summary for ticker, summary in results}

