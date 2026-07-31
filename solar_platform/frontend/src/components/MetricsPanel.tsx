import React from 'react';
import { Award, CheckCircle2, AlertCircle } from 'lucide-react';
import { MetricPair, BlockSummary } from '../api/client';

interface Props {
  metrics: Record<string, MetricPair> | null;
  selectedBlock: BlockSummary;
}

export const MetricsPanel: React.FC<Props> = ({ metrics, selectedBlock }) => {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const metricItem = metrics[selectedBlock.block_id] || metrics[selectedBlock.semantic_label.toLowerCase()] || Object.values(metrics)[0];
  if (!metricItem) return null;

  const r2 = metricItem.r2;
  const mae = metricItem.mae;

  return (
    <div className="ui-card animate-fade-in" style={{ padding: '1.75rem', marginBottom: '2.5rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Award color="#16A34A" size={22} />
          <h3 className="text-card-title" style={{ fontSize: '1.25rem', margin: 0, color: '#0F172A' }}>
            Surrogate Model Performance & Accuracy — {selectedBlock.human_label}
          </h3>
        </div>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#16A34A', backgroundColor: '#DCFCE7', padding: '0.25rem 0.65rem', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <CheckCircle2 size={14} /> Fitted Model Active
        </span>
      </div>

      {/* Metric Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem' }}>
        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #16A34A' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Coefficient of Determination (R²)
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: r2 > 0.9 ? '#16A34A' : '#0F172A' }}>
            {r2.toFixed(4)}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #0284C7' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Mean Absolute Error (MAE)
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0F172A' }}>
            {mae.toFixed(4)} {selectedBlock.target_unit}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Model Feature Set X
          </div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0F172A', marginTop: '0.4rem' }}>
            {[selectedBlock.axis_name, ...selectedBlock.parameters].join(', ')}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Target Output Variable y
          </div>
          <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#0284C7', marginTop: '0.4rem' }}>
            {selectedBlock.target_name} ({selectedBlock.target_unit})
          </div>
        </div>
      </div>
    </div>
  );
};
