import React, { useState, useEffect } from 'react';
import { predictPoint, BlockSummary } from '../api/client';
import { Calculator, Play, CheckCircle2 } from 'lucide-react';

interface Props {
  fileId: string;
  sheetName: string;
  selectedBlock: BlockSummary;
  parameterValues: number[];
  xRanges: Record<string, { min: number; max: number }>;
}

export const PointPredictor: React.FC<Props> = ({
  fileId,
  sheetName,
  selectedBlock,
  parameterValues,
  xRanges
}) => {
  const [xVal, setXVal] = useState<string>('0.5');
  const [paramInputs, setParamInputs] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    // Initialize parameter input controls
    const initParams: Record<string, string> = {};
    if (selectedBlock.parameters && selectedBlock.parameters.length > 0) {
      selectedBlock.parameters.forEach(p => {
        initParams[p] = parameterValues[0]?.toString() || '100';
      });
    }
    setParamInputs(initParams);

    // Set reasonable default X value
    if (selectedBlock.axis_name.toLowerCase().includes('wavelength')) {
      setXVal('500');
    } else {
      setXVal('0.5');
    }
    setPrediction(null);
  }, [selectedBlock, parameterValues]);

  const handleInputChange = (pName: string, val: string) => {
    setParamInputs(prev => ({ ...prev, [pName]: val }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const parsedParamVals: Record<string, number> = {};
      Object.entries(paramInputs).forEach(([k, v]) => {
        parsedParamVals[k] = parseFloat(v) || 0;
      });

      const res = await predictPoint(
        fileId,
        sheetName,
        selectedBlock.semantic_label,
        parseFloat(xVal) || 0,
        undefined,
        selectedBlock.block_id,
        parsedParamVals
      );
      setPrediction(res.predicted_value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const xRange = xRanges[selectedBlock.block_id] || xRanges[selectedBlock.semantic_label.toLowerCase()] || { min: 0, max: 1200 };

  return (
    <div className="ui-card animate-fade-in" style={{ marginBottom: '2.5rem', padding: '1.75rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.8rem' }}>
        <Calculator color="#0284C7" size={22} />
        <h3 className="text-card-title" style={{ fontSize: '1.25rem', margin: 0, color: '#0F172A' }}>
          Live Point Surrogate Prediction Calculator
        </h3>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        {/* Independent Axis Input */}
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', fontWeight: 600 }}>
            {selectedBlock.axis_name} ({selectedBlock.axis_unit})
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
          <span style={{ fontSize: '0.75rem', color: '#64748B' }}>Range: [{xRange.min}, {xRange.max}]</span>
        </div>

        {/* Dynamic Parameter Inputs */}
        {selectedBlock.parameters.map((pName) => (
          <div key={pName}>
            <label style={{ display: 'block', fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', fontWeight: 600 }}>
              {pName} Input
            </label>
            <input
              type="number"
              step="any"
              value={paramInputs[pName] || ''}
              onChange={(e) => handleInputChange(pName, e.target.value)}
              placeholder="e.g. 100"
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
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pt: '1rem', borderTop: '1px solid #F1F5F9' }}>
        <button
          className="btn-primary"
          onClick={handlePredict}
          disabled={loading}
          style={{ padding: '0.75rem 1.6rem', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Play size={16} />
          <span>{loading ? 'Evaluating Model...' : 'Calculate Single Point Prediction'}</span>
        </button>

        {prediction !== null && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', backgroundColor: '#F0FDF4', border: '1px solid #BBF7D0', padding: '0.6rem 1.2rem', borderRadius: '8px' }}>
            <CheckCircle2 color="#16A34A" size={20} />
            <div>
              <span style={{ fontSize: '0.8rem', color: '#166534', fontWeight: 600 }}>Predicted {selectedBlock.target_name}: </span>
              <strong style={{ fontSize: '1.25rem', color: '#15803D', fontWeight: 800 }}>
                {prediction.toFixed(4)} {selectedBlock.target_unit}
              </strong>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
