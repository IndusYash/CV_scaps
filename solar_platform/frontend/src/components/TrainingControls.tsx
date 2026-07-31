import React, { useState } from 'react';
import { trainModels, TrainResponse, BlockSummary } from '../api/client';
import { Cpu, CheckCircle2, BarChart2, Zap } from 'lucide-react';

interface Props {
  fileId: string;
  sheetName: string;
  selectedBlock: BlockSummary;
  onTrainingComplete: (results: TrainResponse) => void;
}

export const TrainingControls: React.FC<Props> = ({
  fileId,
  sheetName,
  selectedBlock,
  onTrainingComplete,
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [nEstimators, setNEstimators] = useState<number>(200);

  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await trainModels(fileId, sheetName, selectedBlock.block_id, testSize, nEstimators);
      onTrainingComplete(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ui-card animate-fade-in" style={{ padding: '1.75rem', marginBottom: '2.5rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.8rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Cpu color="#0284C7" size={22} />
          <h3 className="text-card-title" style={{ fontSize: '1.25rem', margin: 0, color: '#0F172A' }}>
            Random Forest Surrogate Model Fitting — {selectedBlock.human_label}
          </h3>
        </div>
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#0284C7', backgroundColor: '#F0F9FF', padding: '0.25rem 0.65rem', borderRadius: '12px' }}>
          Target: {selectedBlock.target_name} ({selectedBlock.target_unit})
        </span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', fontWeight: 600 }}>
            Validation Split Ratio (Test Size)
          </label>
          <select
            value={testSize}
            onChange={(e) => setTestSize(parseFloat(e.target.value))}
            style={{ width: '100%', padding: '0.65rem', borderRadius: '8px', border: '1px solid #CBD5E1', color: '#0F172A', fontWeight: 600 }}
          >
            <option value={0.1}>10% Test / 90% Train</option>
            <option value={0.2}>20% Test / 80% Train</option>
            <option value={0.3}>30% Test / 70% Train</option>
          </select>
        </div>

        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', color: '#475569', marginBottom: '0.4rem', fontWeight: 600 }}>
            Number of Decision Trees (n_estimators)
          </label>
          <select
            value={nEstimators}
            onChange={(e) => setNEstimators(parseInt(e.target.value))}
            style={{ width: '100%', padding: '0.65rem', borderRadius: '8px', border: '1px solid #CBD5E1', color: '#0F172A', fontWeight: 600 }}
          >
            <option value={100}>100 Decision Trees</option>
            <option value={200}>200 Decision Trees (Recommended)</option>
            <option value={500}>500 Decision Trees</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: '1rem', borderTop: '1px solid #F1F5F9' }}>
        <button
          className="btn-primary"
          onClick={handleTrain}
          disabled={loading}
          style={{ padding: '0.8rem 1.6rem', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <Zap size={18} />
          <span>{loading ? 'Fitting Random Forest Surrogate Model...' : 'Train Surrogate Model'}</span>
        </button>
      </div>
    </div>
  );
};
