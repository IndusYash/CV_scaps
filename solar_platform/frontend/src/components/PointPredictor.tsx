import React, { useState } from 'react';
import { predictPoint } from '../api/client';

interface Props {
  fileId: string;
  sheetName: string;
  availableCurves: string[];
  parameterName: string;
  parameterValues: number[];
  xRanges: Record<string, { min: number; max: number }>;
}

export const PointPredictor: React.FC<Props> = ({
  fileId,
  sheetName,
  availableCurves,
  parameterName,
  parameterValues,
  xRanges
}) => {
  const [curveType, setCurveType] = useState<string>(availableCurves[0] || 'jv');
  const [xVal, setXVal] = useState<string>('0.5');
  const [paramVal, setParamVal] = useState<string>(parameterValues[0]?.toString() || '0');
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await predictPoint(fileId, sheetName, curveType, parseFloat(xVal), parseFloat(paramVal));
      setPrediction(res.predicted_value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurveLabel = (c: string) => {
    if (c.startsWith('block_')) {
      const parts = c.split('_');
      return parts.map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' ');
    }
    return c.toUpperCase();
  };

  const isWavelength = curveType.toLowerCase().includes('qe') || curveType.toLowerCase().includes('responsivity') || curveType.toLowerCase().includes('absorption');
  const xLabel = isWavelength ? 'Wavelength (nm)' : 'Voltage (V)';

  const xRange = xRanges[curveType] || Object.values(xRanges)[0] || { min: 0, max: 1 };
  const isSingleCurve = parameterValues.length === 0;

  return (
    <div className="ui-card" style={{ marginBottom: '2.5rem', padding: '1.75rem', backgroundColor: '#FFFFFF' }}>
      <h3 className="text-card-title" style={{ fontSize: '1.25rem', marginBottom: '1.25rem', borderLeft: '4px solid var(--color-accent)', paddingLeft: '0.8rem' }}>
        Live Point Prediction Calculator
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
            Target Curve Output
          </label>
          <select
            value={curveType}
            onChange={(e) => setCurveType(e.target.value)}
            style={{
              width: '100%',
              padding: '0.65rem 0.9rem',
              borderRadius: '8px',
              border: '1px solid #CBD5E1',
              backgroundColor: '#FFFFFF',
              color: '#0F172A',
              fontWeight: 500,
              fontSize: '0.9rem'
            }}
          >
            {availableCurves.map(c => (
              <option key={c} value={c}>{formatCurveLabel(c)}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
            {xLabel}
          </label>
          <input
            type="number"
            step="0.01"
            min={xRange.min}
            max={xRange.max}
            value={xVal}
            onChange={(e) => setXVal(e.target.value)}
            style={{
              width: '100%',
              padding: '0.65rem 0.9rem',
              borderRadius: '8px',
              border: '1px solid #CBD5E1',
              backgroundColor: '#FFFFFF',
              color: '#0F172A',
              fontWeight: 600,
              fontSize: '0.9rem'
            }}
          />
        </div>

        {!isSingleCurve && (
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
              {parameterName} Input
            </label>
            <select
              value={paramVal}
              onChange={(e) => setParamVal(e.target.value)}
              style={{
                width: '100%',
                padding: '0.65rem 0.9rem',
                borderRadius: '8px',
                border: '1px solid #CBD5E1',
                backgroundColor: '#FFFFFF',
                color: '#0F172A',
                fontWeight: 500,
                fontSize: '0.9rem'
              }}
            >
              {parameterValues.map(v => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <button
          className="btn-primary"
          onClick={handlePredict}
          disabled={loading}
          style={{ padding: '0.7rem 1.4rem' }}
        >
          {loading ? 'Evaluating Model...' : 'Calculate Prediction'}
        </button>

        {prediction !== null && (
          <div style={{ padding: '0.6rem 1.2rem', borderRadius: '8px', backgroundColor: 'var(--color-bg)', border: '1px solid var(--color-border)', fontSize: '0.95rem' }}>
            Predicted Output Value: <strong style={{ color: 'var(--color-accent)', fontSize: '1.1rem' }}>{prediction.toFixed(4)}</strong>
          </div>
        )}
      </div>
    </div>
  );
};
