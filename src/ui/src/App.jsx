import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [trades, setTrades] = useState([]);
  const [brief, setBrief] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRecommendations();
    fetchBrief();
  }, []);

  const fetchRecommendations = async () => {
    try {
      const response = await axios.get(`${API_URL}/recommendations/latest`);
      setTrades(response.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
      setError('Failed to load recommendations');
      setLoading(false);
    }
  };

  const fetchBrief = async () => {
    try {
      const response = await axios.get(`${API_URL}/brief/latest`);
      setBrief(response.data.brief);
    } catch (error) {
      console.error('Error fetching brief:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Swing Trading Recommendations</h1>
        <p>AI-Powered Trade Recommendations for S&P 500 Stocks</p>
      </header>

      <main>
        <section className="brief-section">
          <h2>Daily Brief</h2>
          <div className="brief-content">
            {brief || 'No brief available'}
          </div>
        </section>

        <section className="trades-section">
          <h2>Top Trades ({trades.length})</h2>
          {trades.length === 0 ? (
            <p>No recommendations available for today.</p>
          ) : (
            <table className="trades-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th>Side</th>
                  <th>Target</th>
                  <th>Stop</th>
                  <th>Probability</th>
                  <th>Priority</th>
                  <th>Size</th>
                  <th>Rationale</th>
                </tr>
              </thead>
              <tbody>
                {trades.map((trade, idx) => (
                  <tr key={idx}>
                    <td className="ticker">{trade.ticker}</td>
                    <td className={`side ${trade.side}`}>{trade.side.toUpperCase()}</td>
                    <td className="target">{(trade.target_pct * 100).toFixed(1)}%</td>
                    <td className="stop">{(trade.stop_pct * 100).toFixed(1)}%</td>
                    <td className="probability">{(trade.probability * 100).toFixed(1)}%</td>
                    <td className="priority">{trade.priority_score.toFixed(2)}</td>
                    <td className="size">{(trade.position_size_pct * 100).toFixed(1)}%</td>
                    <td className="rationale">
                      <ul>
                        {trade.rationale.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;

