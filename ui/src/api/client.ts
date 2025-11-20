import axios from 'axios';
import type {
  TradeRecommendation,
  DailyBrief,
  Explanation,
  BacktestResults,
  PaperTradingMetrics,
  TwinsList,
  TwinInfo,
  RegimeInfo,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health check
  health: async (): Promise<{ status: string }> => {
    const response = await client.get('/health');
    return response.data;
  },

  // Recommendations
  getLatestRecommendations: async (): Promise<TradeRecommendation[]> => {
    const response = await client.get('/recommendations/latest');
    return response.data;
  },

  getRecommendationsByDate: async (date: string): Promise<TradeRecommendation[]> => {
    const response = await client.get(`/recommendations/${date}`);
    return response.data;
  },

  // Daily Brief
  getDailyBrief: async (): Promise<DailyBrief> => {
    const response = await client.get('/brief/latest');
    return response.data;
  },

  // Explanation
  explainRecommendation: async (ticker: string, date?: string): Promise<Explanation> => {
    const params = date ? { date } : {};
    const response = await client.get(`/explain/${ticker}`, { params });
    return response.data;
  },

  // Performance
  getBacktestResults: async (): Promise<BacktestResults> => {
    const response = await client.get('/performance/backtest');
    return response.data;
  },

  getPaperTradingPerformance: async (): Promise<PaperTradingMetrics> => {
    const response = await client.get('/performance/paper');
    return response.data;
  },

  // Digital Twins
  listTwins: async (): Promise<TwinsList> => {
    const response = await client.get('/models/twins');
    return response.data;
  },

  getTwinInfo: async (ticker: string): Promise<TwinInfo> => {
    const response = await client.get(`/models/twins/${ticker}`);
    return response.data;
  },

  // Regime
  getTickerRegime: async (ticker: string, date?: string): Promise<RegimeInfo> => {
    const params = date ? { date } : {};
    const response = await client.get(`/predictions/${ticker}/regime`, { params });
    return response.data;
  },
};


