export interface TradeRecommendation {
  ticker: string;
  side: string;
  target_pct: number;
  stop_pct: number;
  probability: number;
  priority_score: number;
  position_size_pct: number;
  rationale: string[];
}

export interface DailyBrief {
  date: string;
  market_context: {
    date: string;
    num_recommendations: number;
    spy_close: number | null;
  };
  brief: string;
  trades: TradeRecommendation[];
}

export interface Explanation {
  ticker: string;
  date: string;
  explanation: string;
  features: {
    expected_return: number;
    probability: number;
  };
}

export interface BacktestResults {
  period: string;
  metrics: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    num_trades: number;
  };
  note?: string;
}

export interface PaperTradingMetrics {
  [key: string]: any;
}

export interface TwinInfo {
  ticker?: string;
  available?: boolean;
  alpha?: number;
  num_parameters?: number;
  characteristics?: Record<string, any>;
}

export interface TwinsList {
  total_twins: number;
  available_twins: number;
  twins: Array<{
    ticker: string;
    available: boolean;
    [key: string]: any;
  }>;
  registry_status?: any;
}

export interface RegimeInfo {
  ticker: string;
  date: string;
  regime_id: number;
  regime_name: string;
  trading_strategy: string;
}

