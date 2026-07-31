import React from 'react';
import { Award, CheckCircle2, AlertTriangle, AlertCircle } from 'lucide-react';
import { MetricPair } from '../api/client';

interface Props {
  metrics: Record<string, MetricPair> | null;
}

export const MetricsPanel: React.FC<Props> = ({ metrics }) => {
  if (!metrics) return null;

  const curveLabels: Record<string, string> = {
    jv: 'J-V Characteristic Curve',
    qe: 'Quantum Efficiency (QE)',
    responsivity: 'Spectral Responsivity',
    detectivity: 'Detectivity (D*)',
    absorption: 'Optical Absorption'
  };

  const formatMetricKey = (key: string) => {
    const kLower = key.toLowerCase();
    if (curveLabels[kLower]) return curveLabels[kLower];
    if (key.startsWith('block_')) {
      const parts = key.split('_');
      return parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
    }
    return key.toUpperCase();
  };

  const getR2Class = (r2: number) => {
    if (r2 >= 0.99) return 'excel';
    if (r2 >= 0.95) return 'good';
    return 'review';
  };

  const getR2Icon = (r2: number) => {
    if (r2 >= 0.99) return <CheckCircle2 size={16} color="var(--accent-mint)" />;
    if (r2 >= 0.95) return <AlertTriangle size={16} color="var(--accent-gold)" />;
    return <AlertCircle size={16} color="var(--accent-coral)" />;
  };

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.2rem' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '3px solid var(--accent-mint)', paddingLeft: '0.8rem' }}>
          <Award color="var(--accent-mint)" size={22} />
          Surrogate Model Validation Metrics
        </h3>
        <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          Evaluated on 20% holdout test dataset
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1.2rem' }}>
        {Object.entries(metrics).map(([key, pair]) => {
          const r2Class = getR2Class(pair.r2);
          return (
            <div key={key} className="glass-card" style={{ padding: '1.4rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.8rem' }}>
                <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)' }}>
                  {formatMetricKey(key)}
                </span>
                {getR2Icon(pair.r2)}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.8rem' }}>
                <div className="metric-card-container">
                  <div className={`metric-val ${r2Class}`}>
                    {pair.r2.toFixed(4)}
                  </div>
                  <div className="metric-lbl">R² Score</div>
                </div>

                <div className="metric-card-container">
                  <div className="metric-val" style={{ color: 'var(--text-main)', fontFamily: 'var(--font-mono)' }}>
                    {pair.mae.toFixed(4)}
                  </div>
                  <div className="metric-lbl">MAE Loss</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
