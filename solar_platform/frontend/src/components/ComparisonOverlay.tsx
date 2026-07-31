import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Layers } from 'lucide-react';
import { predictCurveOverlay, CurvePredictResponse } from '../api/client';

interface Props {
  fileId: string;
  sheetName: string;
  availableCurves: string[];
  parameterName: string;
  parameterValues: number[];
}

export const ComparisonOverlay: React.FC<Props> = ({
  fileId,
  sheetName,
  availableCurves,
  parameterName,
  parameterValues
}) => {
  if (parameterValues.length === 0) return null;

  const [curveType, setCurveType] = useState<string>(availableCurves[0] || 'jv');
  const [paramVal, setParamVal] = useState<number>(parameterValues[0] || 0);
  const [overlayData, setOverlayData] = useState<CurvePredictResponse | null>(null);

  const fetchOverlay = async () => {
    try {
      const data = await predictCurveOverlay(fileId, sheetName, curveType, paramVal);
      setOverlayData(data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    if (fileId && sheetName) {
      fetchOverlay();
    }
  }, [curveType, paramVal, fileId, sheetName]);

  const formatCurveLabel = (c: string) => {
    if (c.startsWith('block_')) {
      const parts = c.split('_');
      return parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
    }
    return c.toUpperCase();
  };

  const isWavelength = curveType.toLowerCase().includes('qe') || curveType.toLowerCase().includes('responsivity') || curveType.toLowerCase().includes('absorption');
  const xLabel = isWavelength ? 'Wavelength (nm)' : 'Voltage (V)';

  return (
    <div className="ui-card" style={{ marginBottom: '2.5rem', padding: '1.75rem', backgroundColor: '#FFFFFF' }}>
      <h3 className="text-card-title" style={{ fontSize: '1.25rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '4px solid var(--color-accent)', paddingLeft: '0.8rem' }}>
        <Layers color="var(--color-accent)" size={22} />
        Curve Comparison (SCAPS Data vs ML Model Prediction)
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
            Select Curve Type
          </label>
          <select
            value={curveType}
            onChange={(e) => setCurveType(e.target.value)}
            style={{
              width: '100%',
              padding: '0.6rem 0.8rem',
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              backgroundColor: '#FFFFFF',
              color: 'var(--color-primary)'
            }}
          >
            {availableCurves.map(c => (
              <option key={c} value={c}>{formatCurveLabel(c)}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
            Select {parameterName} Value
          </label>
          <select
            value={paramVal}
            onChange={(e) => setParamVal(parseFloat(e.target.value))}
            style={{
              width: '100%',
              padding: '0.6rem 0.8rem',
              borderRadius: '8px',
              border: '1px solid var(--color-border)',
              backgroundColor: '#FFFFFF',
              color: 'var(--color-primary)'
            }}
          >
            {parameterValues.map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      </div>

      {overlayData && (
        <Plot
          data={[
            {
              x: overlayData.actual_x,
              y: overlayData.actual_y,
              mode: 'markers',
              name: `Actual SCAPS Data (${parameterName}=${paramVal})`,
              marker: { color: 'var(--color-accent)', size: 6 }
            },
            {
              x: overlayData.pred_x,
              y: overlayData.pred_y,
              mode: 'lines',
              name: `ML Model Prediction (${parameterName}=${paramVal})`,
              line: { color: 'var(--color-secondary)', width: 2.5, dash: 'solid' }
            }
          ]}
          layout={{
            title: { text: `Ground-Truth Data vs Random Forest Surrogate (${formatCurveLabel(curveType)})`, font: { color: '#0F172A', size: 15 } },
            paper_bgcolor: '#FFFFFF',
            plot_bgcolor: '#FFFFFF',
            font: { color: '#334155', size: 12 },
            xaxis: { title: { text: xLabel }, gridcolor: '#E2E8F0', linecolor: '#CBD5E1' },
            yaxis: { title: { text: 'Response Value' }, gridcolor: '#E2E8F0', linecolor: '#CBD5E1' },
            legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0.9)' },
            margin: { l: 65, r: 40, t: 50, b: 55 },
            autosize: true,
            height: 400
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ responsive: true, displayModeBar: true, displaylogo: false }}
        />
      )}
    </div>
  );
};
