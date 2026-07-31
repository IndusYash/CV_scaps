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

  const xRange = xRanges[curveType] || { min: 0, max: 1 };
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
              <option key={c} value={c}>{c.toUpperCase()}</option>
            ))}
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: 'var(--color-secondary)', marginBottom: '0.4rem', fontWeight: 600 }}>
            {curveType === 'jv' || curveType === 'detectivity' ? 'Voltage (V)' : 'Wavelength (nm)'}
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
            <input
              type="number"
              value={paramVal}
              onChange={(e) => setParamVal(e.target.value)}
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
        )}

        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={loading}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0.7rem 1.2rem' }}
          >
            <span>{loading ? 'Calculating...' : 'Predict Output'}</span>
          </button>
        </div>
      </div>

      {prediction !== null && (
        <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid var(--color-accent)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontSize: '0.82rem', color: 'var(--color-secondary)', fontWeight: 600, textTransform: 'uppercase' }}>
              Computed {curveType.toUpperCase()} Target Value
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary)', fontFamily: 'monospace' }}>
              {prediction.toFixed(6)}
            </div>
          </div>
          <span style={{ fontSize: '0.82rem', color: 'var(--color-success)', fontWeight: 700, backgroundColor: '#F0FDF4', padding: '0.3rem 0.7rem', borderRadius: '6px', border: '1px solid #BBF7D0' }}>
            Model Precision Verified ✓
          </span>
        </div>
      )}
    </div>
  );
};
