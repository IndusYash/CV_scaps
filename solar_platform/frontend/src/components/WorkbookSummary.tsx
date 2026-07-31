import React from 'react';
import { Layers, Award, CheckCircle2, ArrowRight, BarChart2, AlertTriangle } from 'lucide-react';
import { UploadResponse, TrainResponse, BlockSummary } from '../api/client';

interface Props {
  uploadData: UploadResponse;
  trainingResults: Record<string, TrainResponse>;
  onSelectSheetAndBlock: (sheetName: string, block: BlockSummary) => void;
  onTrainAll: () => void;
  loadingTrainAll: boolean;
}

export const WorkbookSummary: React.FC<Props> = ({
  uploadData,
  trainingResults,
  onSelectSheetAndBlock,
  onTrainAll,
  loadingTrainAll
}) => {
  const sheets = uploadData.sheets || [];
  const trainedSheetNames = Object.keys(trainingResults);
  const totalSheets = sheets.length;
  const totalBlocks = sheets.reduce((acc, s) => acc + (s.blocks ? s.blocks.length : 0), 0);
  const trainedCount = trainedSheetNames.length;

  let totalR2 = 0;
  let countR2 = 0;

  trainedSheetNames.forEach(sName => {
    const res = trainingResults[sName];
    if (res && res.metrics) {
      Object.values(res.metrics).forEach(m => {
        if (m.r2 !== undefined) {
          totalR2 += m.r2;
          countR2 += 1;
        }
      });
    }
  });

  const avgR2 = countR2 > 0 ? (totalR2 / countR2).toFixed(4) : 'N/A';

  return (
    <div className="animate-fade-in" style={{ width: '100%' }}>
      {/* Overview Header Banner */}
      <div className="ui-card" style={{ padding: '2rem', marginBottom: '2rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0284C7', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.3rem' }}>
              SCAPS Multi-Experiment Analytics Dashboard
            </div>
            <h2 className="text-card-title" style={{ fontSize: '1.6rem', marginBottom: '0.4rem', color: '#0F172A' }}>
              Workbook Experiment Hierarchy
            </h2>
            <p style={{ color: '#475569', margin: 0, fontSize: '0.92rem' }}>
              Discovered experimental blocks, physical parameter sweeps, and surrogate models across all workbook worksheets.
            </p>
          </div>

          {trainedCount < totalSheets && (
            <button className="btn-primary" onClick={onTrainAll} disabled={loadingTrainAll} style={{ padding: '0.85rem 1.5rem', fontSize: '0.95rem' }}>
              <BarChart2 size={18} />
              <span>{loadingTrainAll ? 'Batch Fitting All Worksheets...' : `Train All ${totalSheets} Worksheets`}</span>
            </button>
          )}
        </div>

        {/* Overview Metric Stat Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.25rem' }}>
          <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.2rem' }}>
              Total Worksheets
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0F172A' }}>
              {totalSheets}
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.2rem' }}>
              Discovered Experiments / Blocks
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0284C7' }}>
              {totalBlocks} Blocks
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.2rem' }}>
              Fitted Model Sessions
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: trainedCount === totalSheets ? '#16A34A' : '#0F172A' }}>
              {trainedCount} / {totalSheets}
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#F8FAFC', borderRadius: '10px', border: '1px solid #E2E8F0' }}>
            <div style={{ fontSize: '0.8rem', color: '#64748B', fontWeight: 600, marginBottom: '0.2rem' }}>
              Average Model R² Accuracy
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#0F172A' }}>
              {avgR2 !== 'N/A' ? avgR2 : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Sheet & Block Detailed Grid */}
      <h3 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#0F172A', marginBottom: '1.25rem' }}>
        Discovered Experimental Blocks by Sheet
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
        {sheets.map((s) => {
          const blocks = s.blocks || [];
          const isTrained = trainedSheetNames.includes(s.sheet_name);

          return (
            <div key={s.sheet_name} className="ui-card" style={{ padding: '1.5rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid #F1F5F9', paddingBottom: '0.6rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Layers size={18} color="#0284C7" />
                  <h4 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: '#0F172A' }}>
                    Sheet: '{s.sheet_name}'
                  </h4>
                </div>
                {isTrained && (
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#16A34A', backgroundColor: '#DCFCE7', padding: '0.2rem 0.5rem', borderRadius: '6px' }}>
                    Fitted
                  </span>
                )}
              </div>

              <div style={{ fontSize: '0.82rem', color: '#64748B', marginBottom: '1rem' }}>
                Dimensions: {s.row_count} rows × {s.col_count} cols | {blocks.length} Discovered Block{blocks.length > 1 ? 's' : ''}
              </div>

              {/* Block Cards List */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                {blocks.map((b) => (
                  <div
                    key={b.block_id}
                    onClick={() => onSelectSheetAndBlock(s.sheet_name, b)}
                    style={{
                      padding: '0.9rem',
                      borderRadius: '8px',
                      border: '1px solid #E2E8F0',
                      backgroundColor: b.category === 'SUMMARY' ? '#FEFCE8' : '#F8FAFC',
                      cursor: 'pointer',
                      transition: 'all 0.15s ease'
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.borderColor = '#0284C7')}
                    onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#E2E8F0')}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                      <strong style={{ fontSize: '0.92rem', color: '#0F172A' }}>
                        {b.human_label}
                      </strong>
                      <ArrowRight size={16} color="#0284C7" />
                    </div>

                    <div style={{ fontSize: '0.8rem', color: '#475569', display: 'flex', flexWrap: 'wrap', gap: '0.8rem' }}>
                      <span><strong>Axis:</strong> {b.axis_name} ({b.axis_unit})</span>
                      <span><strong>Target:</strong> {b.target_name} ({b.target_unit})</span>
                    </div>

                    {b.parameters.length > 0 && (
                      <div style={{ fontSize: '0.78rem', color: '#0284C7', fontWeight: 600, marginTop: '0.3rem' }}>
                        Swept Parameters: {b.parameters.join(', ')}
                      </div>
                    )}

                    {b.warnings.length > 0 && (
                      <div style={{ fontSize: '0.75rem', color: '#D97706', marginTop: '0.3rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                        <AlertTriangle size={12} /> Physics Warning: {b.warnings[0]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
