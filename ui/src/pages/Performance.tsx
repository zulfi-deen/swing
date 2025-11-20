import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { BacktestResults, PaperTradingMetrics } from '../api/types';
import MainLayout from '../layouts/MainLayout';
import Card from '../components/Card';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export default function Performance() {
  const [backtestResults, setBacktestResults] = useState<BacktestResults | null>(null);
  const [paperMetrics, setPaperMetrics] = useState<PaperTradingMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPerformanceData();
  }, []);

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [backtest, paper] = await Promise.all([
        api.getBacktestResults(),
        api.getPaperTradingPerformance(),
      ]);
      setBacktestResults(backtest);
      setPaperMetrics(paper);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load performance data');
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

  const metricsData = backtestResults?.metrics
    ? [
        {
          name: 'Total Return',
          value: backtestResults.metrics.total_return,
          format: (v: number) => `${(v * 100).toFixed(2)}%`,
        },
        {
          name: 'Sharpe Ratio',
          value: backtestResults.metrics.sharpe_ratio,
          format: (v: number) => v.toFixed(2),
        },
        {
          name: 'Max Drawdown',
          value: backtestResults.metrics.max_drawdown,
          format: (v: number) => `${(v * 100).toFixed(2)}%`,
        },
        {
          name: 'Win Rate',
          value: backtestResults.metrics.win_rate,
          format: (v: number) => `${(v * 100).toFixed(2)}%`,
        },
        {
          name: 'Num Trades',
          value: backtestResults.metrics.num_trades,
          format: (v: number) => v.toString(),
        },
      ]
    : [];

  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Performance</h1>
            <p className="mt-2 text-sm text-gray-600">
              Backtest and paper trading metrics
            </p>
          </div>
          <button
            onClick={loadPerformanceData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>

        {error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-yellow-800">
            {error}
          </div>
        )}

        {/* Backtest Results */}
        <Card title="Backtest Results">
          {backtestResults ? (
            <div className="space-y-6">
              <div className="text-sm text-gray-600">
                Period: {backtestResults.period}
              </div>
              {backtestResults.note && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-800">
                  {backtestResults.note}
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                {metricsData.map((metric) => (
                  <MetricCard
                    key={metric.name}
                    name={metric.name}
                    value={metric.value}
                    format={metric.format}
                  />
                ))}
              </div>
              {metricsData.length > 0 && (
                <div className="mt-6">
                  <h4 className="text-lg font-semibold text-gray-900 mb-4">
                    Performance Metrics
                  </h4>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={metricsData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip
                        formatter={(value: number) => {
                          const metric = metricsData.find((m) => m.value === value);
                          return metric ? metric.format(value) : value;
                        }}
                      />
                      <Legend />
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500">No backtest data available</p>
          )}
        </Card>

        {/* Paper Trading Metrics */}
        <Card title="Paper Trading Performance">
          {paperMetrics ? (
            <div className="space-y-4">
              {Object.keys(paperMetrics).length > 0 ? (
                <div className="bg-gray-50 rounded-lg p-4">
                  <pre className="text-sm text-gray-700 overflow-x-auto">
                    {JSON.stringify(paperMetrics, null, 2)}
                  </pre>
                </div>
              ) : (
                <p className="text-gray-500">No paper trading data available</p>
              )}
            </div>
          ) : (
            <p className="text-gray-500">No paper trading data available</p>
          )}
        </Card>
      </div>
    </MainLayout>
  );
}

interface MetricCardProps {
  name: string;
  value: number;
  format: (value: number) => string;
}

function MetricCard({ name, value, format }: MetricCardProps) {
  const isPositive = name === 'Total Return' || name === 'Sharpe Ratio' || name === 'Win Rate';
  const Icon = isPositive ? TrendingUp : TrendingDown;
  const colorClass = isPositive
    ? value >= 0
      ? 'text-green-600'
      : 'text-red-600'
    : 'text-gray-600';

  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm text-gray-600">{name}</p>
        <Icon className={`h-4 w-4 ${colorClass}`} />
      </div>
      <p className={`text-2xl font-bold ${colorClass}`}>{format(value)}</p>
    </div>
  );
}


