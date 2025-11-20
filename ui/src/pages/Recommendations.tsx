import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TradeRecommendation, Explanation } from '../api/types';
import MainLayout from '../layouts/MainLayout';
import Card from '../components/Card';
import Badge from '../components/Badge';
import { TrendingUp, TrendingDown, Info, X } from 'lucide-react';

export default function Recommendations() {
  const [recommendations, setRecommendations] = useState<TradeRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.getLatestRecommendations();
      setRecommendations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async (ticker: string) => {
    if (selectedTicker === ticker && explanation) {
      setSelectedTicker(null);
      setExplanation(null);
      return;
    }

    try {
      setLoadingExplanation(true);
      setSelectedTicker(ticker);
      const data = await api.explainRecommendation(ticker);
      setExplanation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load explanation');
    } finally {
      setLoadingExplanation(false);
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

  if (error && !recommendations.length) {
    return (
      <MainLayout>
        <Card>
          <div className="text-red-600">Error: {error}</div>
        </Card>
      </MainLayout>
    );
  }

  const sortedRecommendations = [...recommendations].sort(
    (a, b) => b.priority_score - a.priority_score
  );

  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Trade Recommendations</h1>
            <p className="mt-2 text-sm text-gray-600">
              {recommendations.length} recommendations available
            </p>
          </div>
          <button
            onClick={loadRecommendations}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>

        {error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800">
            {error}
          </div>
        )}

        <div className="grid gap-4">
          {sortedRecommendations.map((trade, index) => (
            <TradeRecommendationCard
              key={`${trade.ticker}-${index}`}
              trade={trade}
              onExplain={() => handleExplain(trade.ticker)}
              isExpanded={selectedTicker === trade.ticker}
              explanation={selectedTicker === trade.ticker ? explanation : null}
              loadingExplanation={loadingExplanation && selectedTicker === trade.ticker}
            />
          ))}
        </div>

        {recommendations.length === 0 && (
          <Card>
            <p className="text-gray-500 text-center py-8">No recommendations available</p>
          </Card>
        )}
      </div>
    </MainLayout>
  );
}

interface TradeRecommendationCardProps {
  trade: TradeRecommendation;
  onExplain: () => void;
  isExpanded: boolean;
  explanation: Explanation | null;
  loadingExplanation: boolean;
}

function TradeRecommendationCard({
  trade,
  onExplain,
  isExpanded,
  explanation,
  loadingExplanation,
}: TradeRecommendationCardProps) {
  const isLong = trade.side.toLowerCase() === 'long' || trade.side.toLowerCase() === 'buy';
  const SideIcon = isLong ? TrendingUp : TrendingDown;

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-4 flex-1">
            <div
              className={`p-3 rounded-lg ${
                isLong ? 'bg-green-100' : 'bg-red-100'
              }`}
            >
              <SideIcon
                className={`h-6 w-6 ${isLong ? 'text-green-600' : 'text-red-600'}`}
              />
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-3 mb-2">
                <h3 className="text-xl font-bold text-gray-900">{trade.ticker}</h3>
                <Badge variant={isLong ? 'success' : 'danger'}>
                  {trade.side.toUpperCase()}
                </Badge>
                <Badge variant="info">
                  Priority: {trade.priority_score.toFixed(2)}
                </Badge>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Target</p>
                  <p className="font-semibold text-gray-900">
                    {(trade.target_pct * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Stop Loss</p>
                  <p className="font-semibold text-gray-900">
                    {(trade.stop_pct * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Probability</p>
                  <p className="font-semibold text-gray-900">
                    {(trade.probability * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">Position Size</p>
                  <p className="font-semibold text-gray-900">
                    {(trade.position_size_pct * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          </div>
          <button
            onClick={onExplain}
            className="ml-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <Info className="h-4 w-4" />
            <span>{isExpanded ? 'Hide' : 'Explain'}</span>
          </button>
        </div>

        {trade.rationale && trade.rationale.length > 0 && (
          <div className="border-t border-gray-200 pt-4">
            <p className="text-sm font-semibold text-gray-700 mb-2">Rationale:</p>
            <ul className="list-disc list-inside space-y-1 text-sm text-gray-600">
              {trade.rationale.map((reason, idx) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>
        )}

        {isExpanded && (
          <div className="border-t border-gray-200 pt-4">
            {loadingExplanation ? (
              <div className="flex justify-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
              </div>
            ) : explanation ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold text-gray-900">Detailed Explanation</h4>
                  <button
                    onClick={onExplain}
                    className="text-gray-400 hover:text-gray-600"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-700 whitespace-pre-wrap">
                    {explanation.explanation}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-500">Expected Return</p>
                    <p className="font-semibold text-gray-900">
                      {(explanation.features.expected_return * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-500">Probability</p>
                    <p className="font-semibold text-gray-900">
                      {(explanation.features.probability * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No explanation available</p>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}


