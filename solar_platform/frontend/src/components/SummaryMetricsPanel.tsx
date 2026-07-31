import React from 'react';
import { Award, Zap, Activity, ShieldCheck, CheckCircle2 } from 'lucide-react';

interface Props {
  summaryMetrics: Record<string, Record<string, number>>;
  sheetName: string;
}

export const SummaryMetricsPanel: React.FC<Props> = ({ summaryMetrics, sheetName }) => {
  if (!summaryMetrics || Object.keys(summaryMetrics).length === 0) return null;

  const getMetricValue = (keys: string[]): number | null => {
    for (const metricKey of Object.keys(summaryMetrics)) {
      const mkLower = metricKey.toLowerCase();
      if (keys.some(k => mkLower.includes(k.toLowerCase()))) {
        const item = summaryMetrics[metricKey];
        return item.mean ?? item.latest ?? item.max ?? null;
      }
    }
    return null;
  };

  const voc = getMetricValue(['voc']);
  const jsc = getMetricValue(['jsc', 'j_sc']);
  const ff = getMetricValue(['ff']);
  const eta = getMetricValue(['eta', 'pce', 'eff']);
  const vmpp = getMetricValue(['v_mpp', 'vmpp']);
  const jmpp = getMetricValue(['j_mpp', 'jmpp']);

  return (
    <div className="ui-card animate-fade-in" style={{ padding: '1.75rem', marginBottom: '2.5rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Award color="#0284C7" size={22} />
          <h3 className="text-card-title" style={{ fontSize: '1.25rem', margin: 0, color: '#0F172A' }}>
            Photovoltaic Device Performance Summary Metrics — '{sheetName}'
          </h3>
        </div>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#16A34A', backgroundColor: '#DCFCE7', padding: '0.25rem 0.65rem', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
          <CheckCircle2 size={14} /> Direct SCAPS Output
        </span>
      </div>

      <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '1.5rem', lineHeight: 1.5 }}>
        Extracted device figure-of-merit metrics. Note: Summary performance tables are stored as non-training scientific result cards.
      </p>

      {/* Metric Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #0284C7' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Open-Circuit Voltage (Voc)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0F172A' }}>
            {voc !== null ? `${voc.toFixed(4)} V` : 'N/A'}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #0284C7' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Short-Circuit Current (Jsc)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0F172A' }}>
            {jsc !== null ? `${jsc.toFixed(4)} mA/cm²` : 'N/A'}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #0284C7' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Fill Factor (FF)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#0F172A' }}>
            {ff !== null ? `${ff.toFixed(2)} %` : 'N/A'}
          </div>
        </div>

        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0', borderTop: '4px solid #16A34A' }}>
          <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.3rem' }}>
            Power Conversion Eff (η / PCE)
          </div>
          <div style={{ fontSize: '1.6rem', fontWeight: 800, color: '#16A34A' }}>
            {eta !== null ? `${eta.toFixed(2)} %` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Raw Metrics Detail Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ backgroundColor: '#F1F5F9', borderBottom: '2px solid #CBD5E1', textAlign: 'left' }}>
              <th style={{ padding: '0.6rem 0.9rem', color: '#334155', fontWeight: 700 }}>Scalar Metric Name</th>
              <th style={{ padding: '0.6rem 0.9rem', color: '#334155', fontWeight: 700 }}>Mean Value</th>
              <th style={{ padding: '0.6rem 0.9rem', color: '#334155', fontWeight: 700 }}>Max Value</th>
              <th style={{ padding: '0.6rem 0.9rem', color: '#334155', fontWeight: 700 }}>Min Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(summaryMetrics).map(([mName, mVals], idx) => (
              <tr key={mName} style={{ borderBottom: '1px solid #E2E8F0', backgroundColor: idx % 2 === 0 ? '#FFFFFF' : '#F8FAFC' }}>
                <td style={{ padding: '0.55rem 0.9rem', fontWeight: 600, color: '#0F172A' }}>{mName}</td>
                <td style={{ padding: '0.55rem 0.9rem', color: '#334155' }}>{mVals.mean?.toFixed(4) ?? '—'}</td>
                <td style={{ padding: '0.55rem 0.9rem', color: '#334155' }}>{mVals.max?.toFixed(4) ?? '—'}</td>
                <td style={{ padding: '0.55rem 0.9rem', color: '#334155' }}>{mVals.min?.toFixed(4) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
