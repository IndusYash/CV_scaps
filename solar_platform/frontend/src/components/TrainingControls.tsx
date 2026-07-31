import React from 'react';
import { Sliders, Cpu, Play } from 'lucide-react';

interface Props {
  testSize: number;
  setTestSize: (val: number) => void;
  nEstimators: number;
  setNEstimators: (val: number) => void;
  randomState: number;
  setRandomState: (val: number) => void;
  onTrain: () => void;
  loading: boolean;
}

export const TrainingControls: React.FC<Props> = ({
  testSize,
  setTestSize,
  nEstimators,
  setNEstimators,
  randomState,
  setRandomState,
  onTrain,
  loading
}) => {
  return (
    <div className="glass-card" style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.2rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '3px solid var(--accent-cyan)', paddingLeft: '0.8rem' }}>
          <Sliders color="var(--accent-cyan)" size={22} />
          Surrogate Model Hyperparameters
        </h3>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Configures Random Forest Regressor architecture
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.5rem', alignItems: 'flex-end' }}>
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Test Split Ratio</span>
            <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{(testSize * 100).toFixed(0)}%</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.05"
            value={testSize}
            onChange={(e) => setTestSize(parseFloat(e.target.value))}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Random Forest Trees</span>
            <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{nEstimators} Trees</span>
          </div>
          <input
            type="range"
            min="50"
            max="600"
            step="50"
            value={nEstimators}
            onChange={(e) => setNEstimators(parseInt(e.target.value))}
          />
        </div>

        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>Random Seed</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Reproducibility</span>
          </div>
          <input
            type="number"
            value={randomState}
            onChange={(e) => setRandomState(parseInt(e.target.value) || 0)}
            style={{ width: '100%', padding: '0.55rem 0.8rem' }}
          />
        </div>

        <div>
          <button
            className="btn-primary"
            onClick={onTrain}
            disabled={loading}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem' }}
          >
            <Play size={18} />
            {loading ? 'Fitting Regressors...' : 'Train Physics Models'}
          </button>
        </div>
      </div>
    </div>
  );
};
