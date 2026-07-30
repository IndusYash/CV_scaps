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

  const xLabel = curveType === 'jv' || curveType === 'detectivity' ? 'Voltage (V)' : 'Wavelength (nm)';

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
              <option key={c} value={c}>{c.toUpperCase()}</option>
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
              mode: 'lines' as const,
              name: 'Actual SCAPS Data',
              line: { color: '#0F172A', width: 3 }
            },
            {
              x: overlayData.pred_x,
              y: overlayData.pred_y,
              mode: 'lines' as const,
              name: 'ML Predicted Curve',
              line: { color: '#2563EB', dash: 'dash', width: 3 }
            }
          ]}
          layout={{
            title: {
              text: `Curve Overlay Comparison: ${curveType.toUpperCase()} at ${parameterName} = ${paramVal}`,
              font: { color: '#0F172A', family: 'Inter, sans-serif', size: 15, weight: 700 }
            },
            paper_bgcolor: '#FFFFFF',
            plot_bgcolor: '#FFFFFF',
            font: { color: '#334155', family: 'Inter, sans-serif', size: 12 },
            xaxis: {
              title: { text: xLabel, font: { color: '#0F172A', size: 13, weight: 600 } },
              tickfont: { color: '#334155', size: 11 },
              gridcolor: '#E2E8F0',
              linecolor: '#CBD5E1'
            },
            yaxis: {
              title: { text: 'Value', font: { color: '#0F172A', size: 13, weight: 600 } },
              tickfont: { color: '#334155', size: 11 },
              gridcolor: '#E2E8F0',
              linecolor: '#CBD5E1'
            },
            legend: {
              font: { color: '#0F172A', size: 12 },
              bgcolor: 'rgba(255, 255, 255, 0.9)',
              bordercolor: '#E2E8F0',
              borderwidth: 1
            },
            autosize: true,
            height: 420
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ responsive: true, displaylogo: false }}
        />
      )}
    </div>
  );
};
