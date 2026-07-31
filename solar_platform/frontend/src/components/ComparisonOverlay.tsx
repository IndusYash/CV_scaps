import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import { predictCurveOverlay, BlockSummary } from '../api/client';
import { GitCompare, Sparkles } from 'lucide-react';

interface Props {
  fileId: string;
  sheetName: string;
  selectedBlock: BlockSummary;
  parameterValues: number[];
}

export const ComparisonOverlay: React.FC<Props> = ({
  fileId,
  sheetName,
  selectedBlock,
  parameterValues
}) => {
  const [selectedParamVal, setSelectedParamVal] = useState<number>(parameterValues[0] || 1.0);
  const [actualX, setActualX] = useState<number[]>([]);
  const [actualY, setActualY] = useState<number[]>([]);
  const [predX, setPredX] = useState<number[]>([]);
  const [predY, setPredY] = useState<number[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [hasCompared, setHasCompared] = useState<boolean>(false);

  const handleGenerateOverlay = async () => {
    setLoading(true);
    try {
      const res = await predictCurveOverlay(
        fileId,
        sheetName,
        selectedBlock.semantic_label,
        selectedParamVal,
        selectedBlock.block_id
      );
      setActualX(res.actual_x);
      setActualY(res.actual_y);
      setPredX(res.pred_x);
      setPredY(res.pred_y);
      setHasCompared(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const paramLabel = selectedBlock.parameters.length > 0 ? selectedBlock.parameters.join(', ') : 'Parameter';

  return (
    <div className="ui-card animate-fade-in" style={{ marginBottom: '2.5rem', padding: '1.75rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.8rem' }}>
        <GitCompare color="#0284C7" size={22} />
        <h3 className="text-card-title" style={{ fontSize: '1.25rem', margin: 0, color: '#0F172A' }}>
          Actual SCAPS Data vs Random Forest Surrogate Curve Overlay
        </h3>
      </div>

      <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '1.25rem' }}>
        Simultaneously plot ground-truth SCAPS simulation points against continuous Random Forest surrogate predictions.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem', marginBottom: '1.5rem' }}>
        {parameterValues.length > 0 && (
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', fontWeight: 600 }}>
              Select {paramLabel} Value:
            </label>
            <select
              value={selectedParamVal}
              onChange={(e) => setSelectedParamVal(parseFloat(e.target.value))}
              style={{
                padding: '0.65rem 1rem',
                borderRadius: '8px',
                border: '1px solid #CBD5E1',
                backgroundColor: '#FFFFFF',
                color: '#0F172A',
                fontWeight: 600,
                fontSize: '0.9rem'
              }}
            >
              {parameterValues.map(v => (
                <option key={v} value={v}>{paramLabel} = {v}</option>
              ))}
            </select>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={handleGenerateOverlay}
          disabled={loading}
          style={{ padding: '0.7rem 1.4rem', fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '0.5rem', alignSelf: 'flex-end' }}
        >
          <Sparkles size={16} />
          <span>{loading ? 'Generating Trace Comparison...' : 'Generate Actual vs Predicted Curve'}</span>
        </button>
      </div>

      {/* Plotly Figure */}
      {hasCompared && (
        <div style={{ borderRadius: '10px', overflow: 'hidden', border: '1px solid #E2E8F0', padding: '1rem', backgroundColor: '#FFFFFF' }}>
          <Plot
            data={[
              {
                x: actualX,
                y: actualY,
                mode: 'markers',
                name: `Actual SCAPS Data (${paramLabel}=${selectedParamVal})`,
                marker: { color: '#0284C7', size: 7, symbol: 'circle' }
              },
              {
                x: predX,
                y: predY,
                mode: 'lines',
                name: `Surrogate Prediction (${paramLabel}=${selectedParamVal})`,
                line: { color: '#DC2626', width: 2.5, dash: 'solid' }
              }
            ]}
            layout={{
              title: {
                text: `${selectedBlock.human_label} — Overlay Analysis`,
                font: { color: '#0F172A', family: 'Inter, sans-serif', size: 15, weight: 700 }
              },
              paper_bgcolor: '#FFFFFF',
              plot_bgcolor: '#FFFFFF',
              font: { color: '#334155', family: 'Inter, sans-serif', size: 12 },
              xaxis: {
                title: { text: `${selectedBlock.axis_name} (${selectedBlock.axis_unit})`, font: { color: '#0F172A', size: 13, weight: 600 } },
                gridcolor: '#E2E8F0',
                linecolor: '#CBD5E1'
              },
              yaxis: {
                title: { text: `${selectedBlock.target_name} (${selectedBlock.target_unit})`, font: { color: '#0F172A', size: 13, weight: 600 } },
                gridcolor: '#E2E8F0',
                linecolor: '#CBD5E1'
              },
              legend: {
                font: { color: '#0F172A', size: 12 },
                bgcolor: 'rgba(255, 255, 255, 0.95)',
                bordercolor: '#E2E8F0',
                borderwidth: 1
              },
              margin: { l: 65, r: 40, t: 50, b: 55 },
              autosize: true,
              height: 420
            }}
            useResizeHandler={true}
            style={{ width: '100%', height: '100%' }}
            config={{ responsive: true, displayModeBar: true, displaylogo: false }}
          />
        </div>
      )}
    </div>
  );
};
