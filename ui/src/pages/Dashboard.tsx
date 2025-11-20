import { useEffect, useState } from 'react';
import { format } from 'date-fns';
import { api } from '../api/client';
import type { DailyBrief, TradeRecommendation } from '../api/types';
import MainLayout from '../layouts/MainLayout';
import Card from '../components/Card';
import Badge from '../components/Badge';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

export default function Dashboard() {
  const [brief, setBrief] = useState<DailyBrief | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDailyBrief();
  }, []);

  const loadDailyBrief = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getDailyBrief();
      setBrief(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load daily brief');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <MainLayout>
        <div className="flex justify-center items-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      </MainLayout>
    );
  }

  if (error) {
    return (
      <MainLayout>
        <Card>
          <div className="flex items-center text-red-600">
            <AlertCircle className="h-5 w-5 mr-2" />
            <span>Error: {error}</span>
          </div>
        </Card>
      </MainLayout>
    );
  }

  if (!brief) {
    return (
      <MainLayout>
        <Card>
          <p className="text-gray-500">No daily brief available</p>
        </Card>
      </MainLayout>
    );
  }

  const topTrades = brief.trades.slice(0, 5);

  return (
    <MainLayout>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-2 text-sm text-gray-600">
            {brief.date && format(new Date(brief.date), 'EEEE, MMMM d, yyyy')}
          </p>
        </div>

        {/* Market Context */}
        <Card title="Market Context">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <p className="text-sm text-gray-600">Date</p>
              <p className="text-lg font-semibold">
                {brief.date && format(new Date(brief.date), 'MMM d, yyyy')}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Recommendations</p>
              <p className="text-lg font-semibold">{brief.market_context.num_recommendations}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">SPY Close</p>
              <p className="text-lg font-semibold">
                {brief.market_context.spy_close
                  ? `$${brief.market_context.spy_close.toFixed(2)}`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </Card>

        {/* Daily Brief */}
        <Card title="Daily Brief">
          <div className="whitespace-pre-wrap text-gray-700">{brief.brief}</div>
        </Card>

        {/* Top Recommendations */}
        <Card title="Top Recommendations">
          <div className="space-y-4">
            {topTrades.map((trade, index) => (
              <TradeCard key={`${trade.ticker}-${index}`} trade={trade} />
            ))}
          </div>
        </Card>
      </div>
    </MainLayout>
  );
}

function TradeCard({ trade }: { trade: TradeRecommendation }) {
  const isLong = trade.side.toLowerCase() === 'long' || trade.side.toLowerCase() === 'buy';
  const SideIcon = isLong ? TrendingUp : TrendingDown;

  return (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-center space-x-3">
          <div
            className={`p-2 rounded-full ${
              isLong ? 'bg-green-100' : 'bg-red-100'
            }`}
          >
            <SideIcon
              className={`h-5 w-5 ${isLong ? 'text-green-600' : 'text-red-600'}`}
            />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h4 className="text-lg font-semibold text-gray-900">{trade.ticker}</h4>
              <Badge variant={isLong ? 'success' : 'danger'}>
                {trade.side.toUpperCase()}
              </Badge>
            </div>
            <div className="mt-1 flex flex-wrap gap-2 text-sm text-gray-600">
              <span>Target: {(trade.target_pct * 100).toFixed(1)}%</span>
              <span>•</span>
              <span>Stop: {(trade.stop_pct * 100).toFixed(1)}%</span>
              <span>•</span>
              <span>Probability: {(trade.probability * 100).toFixed(1)}%</span>
              <span>•</span>
              <span>Size: {(trade.position_size_pct * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500">Priority Score</p>
          <p className="text-lg font-bold text-blue-600">
            {trade.priority_score.toFixed(2)}
          </p>
        </div>
      </div>
      {trade.rationale && trade.rationale.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs font-semibold text-gray-500 mb-1">Rationale:</p>
          <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
            {trade.rationale.map((reason, idx) => (
              <li key={idx}>{reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

