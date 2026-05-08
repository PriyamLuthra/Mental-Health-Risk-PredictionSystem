import { useState } from 'react';
import { Brain, AlertCircle, CheckCircle2, RefreshCcw } from 'lucide-react';
import './index.css';

function App() {
  const [formData, setFormData] = useState({
    age: 25,
    sleep: 7,
    outside: 2,
    screen: 0,
    physical: 0,
    diet: 0
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? (checked ? 1 : 0) : Number(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err.message || 'Could not connect to the backend. Make sure Flask is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header>
        <h1>MindWell Predictor</h1>
        <p>AI-Powered Mental Health Risk Assessment</p>
      </header>

      <div className="main-content">
        {/* Left Panel: Form */}
        <div className="card form-card">
          <form onSubmit={handleSubmit}>

            <div className="form-group">
              <label>Age (Years)</label>
              <input type="number" name="age" value={formData.age}
                onChange={handleInputChange} min="1" max="120" required />
            </div>

            <div className="form-group">
              <label>Sleep Duration (Hours/Night)</label>
              <input type="number" name="sleep" value={formData.sleep}
                onChange={handleInputChange} step="0.5" min="0" max="24" required />
            </div>

            <div className="form-group">
              <label>Hours Spent Outside Daily</label>
              <input type="number" name="outside" value={formData.outside}
                onChange={handleInputChange} step="0.5" min="0" max="24" required />
            </div>

            <div className="form-group">
              <label>High Screen Time (&gt;4 hrs/day)</label>
              <div className="toggle-group">
                <label className="toggle-switch">
                  <input type="checkbox" name="screen"
                    checked={formData.screen === 1}
                    onChange={handleInputChange} />
                  <span className="slider"></span>
                </label>
                <span>{formData.screen ? 'Yes' : 'No'}</span>
              </div>
            </div>

            <div className="form-group">
              <label>Physical Activity Today</label>
              <div className="toggle-group">
                <label className="toggle-switch">
                  <input type="checkbox" name="physical"
                    checked={formData.physical === 1}
                    onChange={handleInputChange} />
                  <span className="slider"></span>
                </label>
                <span>{formData.physical ? 'Active' : 'Sedentary'}</span>
              </div>
            </div>

            <div className="form-group">
              <label>Healthy Diet Followed</label>
              <div className="toggle-group">
                <label className="toggle-switch">
                  <input type="checkbox" name="diet"
                    checked={formData.diet === 1}
                    onChange={handleInputChange} />
                  <span className="slider"></span>
                </label>
                <span>{formData.diet ? 'Balanced' : 'Irregular'}</span>
              </div>
            </div>

            <button type="submit" className="predict-btn" disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Risk Status'}
            </button>
          </form>
        </div>

        {/* Right Panel: Results */}
        <div className="card results-card">
          {loading ? (
            <div className="results-panel">
              <div className="loader"></div>
              <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Processing lifestyle markers...</p>
            </div>
          ) : result ? (
            <div className="results-panel">
              <div className={`risk-indicator ${result.prediction === 1 ? 'high' : 'low'}`}>
                <span className="status-label">Risk Level</span>
                <span className="status-value">{result.prediction === 1 ? 'Elevated' : 'Optimal'}</span>
                {result.prediction === 1 ? <AlertCircle size={28} /> : <CheckCircle2 size={28} />}
              </div>

              <h2 style={{ marginTop: '1rem' }}>{result.risk_status}</h2>
              <p className="probability-ring" style={{ marginTop: '0.5rem' }}>
                Confidence: <strong>{result.probability}%</strong>
              </p>

              <p style={{ color: 'var(--text-muted)', marginTop: '1.5rem', fontSize: '0.9rem', lineHeight: '1.6' }}>
                {result.prediction === 1
                  ? 'Your lifestyle markers suggest elevated mental health risk. Consider reducing screen time, improving sleep, and staying active.'
                  : 'Great news! Your lifestyle reflects healthy habits. Keep maintaining this balance for optimal well-being.'}
              </p>

              <button onClick={resetForm} style={{
                background: 'none', border: '1px solid rgba(255,255,255,0.1)',
                color: 'var(--text-muted)', marginTop: '2rem', padding: '0.5rem 1.2rem',
                borderRadius: '0.5rem', cursor: 'pointer', display: 'flex',
                alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem'
              }}>
                <RefreshCcw size={14} /> New Assessment
              </button>
            </div>
          ) : error ? (
            <div className="results-panel">
              <AlertCircle size={48} color="var(--danger)" />
              <p style={{ color: 'var(--danger)', marginTop: '1rem', textAlign: 'center' }}>{error}</p>
              <button className="predict-btn" onClick={handleSubmit} style={{ marginTop: '2rem' }}>
                Retry
              </button>
            </div>
          ) : (
            <div className="empty-state">
              <Brain className="empty-state-icon" style={{ width: 64, height: 64, opacity: 0.3 }} />
              <h3 style={{ marginTop: '1rem' }}>Awaiting Input</h3>
              <p style={{ marginTop: '0.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                Fill in the wellness profile on the left to get your AI-powered mental health risk analysis.
              </p>
            </div>
          )}
        </div>
      </div>

      <footer style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
        <p>&#169; 2026 MindWell AI &mdash; For awareness only. Not a medical diagnosis.</p>
      </footer>
    </div>
  );
}

export default App;
