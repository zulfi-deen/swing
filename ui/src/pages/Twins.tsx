import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { TwinsList, TwinInfo } from '../api/types';
import MainLayout from '../layouts/MainLayout';
import Card from '../components/Card';
import Badge from '../components/Badge';
import { Cpu, CheckCircle, XCircle, Info, RefreshCw } from 'lucide-react';

export default function Twins() {
  const [twinsList, setTwinsList] = useState<TwinsList | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [twinInfo, setTwinInfo] = useState<TwinInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTwins();
  }, []);

  const loadTwins = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await api.listTwins();
      setTwinsList(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load digital twins');
    } finally {
      setLoading(false);
    }
  };

  const loadTwinDetails = async (ticker: string) => {
    if (selectedTicker === ticker && twinInfo) {
      setSelectedTicker(null);
      setTwinInfo(null);
      return;
    }

    try {
      setLoadingDetails(true);
      setSelectedTicker(ticker);
      const data = await api.getTwinInfo(ticker);
      setTwinInfo(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load twin details');
    } finally {
      setLoadingDetails(false);
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

  if (error && !twinsList) {
    return (
      <MainLayout>
        <Card>
          <div className="text-red-600">Error: {error}</div>
        </Card>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Digital Twins</h1>
            <p className="mt-2 text-sm text-gray-600">
              {twinsList?.total_twins || 0} total twins,{' '}
              {twinsList?.available_twins || 0} available
            </p>
          </div>
          <button
            onClick={loadTwins}
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

        {twinsList && twinsList.twins.length > 0 ? (
          <div className="grid gap-4">
            {twinsList.twins.map((twin) => (
              <TwinCard
                key={twin.ticker}
                twin={twin}
                onViewDetails={() => loadTwinDetails(twin.ticker)}
                isExpanded={selectedTicker === twin.ticker}
                details={selectedTicker === twin.ticker ? twinInfo : null}
                loadingDetails={loadingDetails && selectedTicker === twin.ticker}
              />
            ))}
          </div>
        ) : (
          <Card>
            <div className="text-center py-8">
              <Cpu className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No digital twins available</p>
            </div>
          </Card>
        )}
      </div>
    </MainLayout>
  );
}

interface TwinCardProps {
  twin: { ticker: string; available: boolean; [key: string]: any };
  onViewDetails: () => void;
  isExpanded: boolean;
  details: TwinInfo | null;
  loadingDetails: boolean;
}

function TwinCard({
  twin,
  onViewDetails,
  isExpanded,
  details,
  loadingDetails,
}: TwinCardProps) {
  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Cpu className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <div className="flex items-center space-x-3">
                <h3 className="text-xl font-bold text-gray-900">{twin.ticker}</h3>
                <Badge variant={twin.available ? 'success' : 'danger'}>
                  {twin.available ? (
                    <>
                      <CheckCircle className="h-3 w-3 inline mr-1" />
                      Available
                    </>
                  ) : (
                    <>
                      <XCircle className="h-3 w-3 inline mr-1" />
                      Unavailable
                    </>
                  )}
                </Badge>
              </div>
            </div>
          </div>
          {twin.available && (
            <button
              onClick={onViewDetails}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
            >
              <Info className="h-4 w-4" />
              <span>{isExpanded ? 'Hide Details' : 'View Details'}</span>
            </button>
          )}
        </div>

        {isExpanded && twin.available && (
          <div className="border-t border-gray-200 pt-4">
            {loadingDetails ? (
              <div className="flex justify-center py-4">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
              </div>
            ) : details ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {details.alpha !== undefined && (
                    <div>
                      <p className="text-sm text-gray-500">Alpha</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {details.alpha.toFixed(4)}
                      </p>
                    </div>
                  )}
                  {details.num_parameters !== undefined && (
                    <div>
                      <p className="text-sm text-gray-500">Parameters</p>
                      <p className="text-lg font-semibold text-gray-900">
                        {details.num_parameters.toLocaleString()}
                      </p>
                    </div>
                  )}
                  {details.available !== undefined && (
                    <div>
                      <p className="text-sm text-gray-500">Status</p>
                      <Badge variant={details.available ? 'success' : 'danger'}>
                        {details.available ? 'Available' : 'Unavailable'}
                      </Badge>
                    </div>
                  )}
                </div>
                {details.characteristics && Object.keys(details.characteristics).length > 0 && (
                  <div>
                    <p className="text-sm font-semibold text-gray-700 mb-2">Characteristics:</p>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <pre className="text-xs text-gray-600 overflow-x-auto">
                        {JSON.stringify(details.characteristics, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No details available</p>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}


