import React from 'react';
import { Layers, Award, CheckCircle2, ArrowRight, BarChart2 } from 'lucide-react';
import { UploadResponse, TrainResponse } from '../api/client';

interface Props {
  uploadData: UploadResponse;
  trainingResults: Record<string, TrainResponse>;
  onSelectSheet: (sheetName: string) => void;
  onTrainAll: () => void;
  loadingTrainAll: boolean;
}

export const WorkbookSummary: React.FC<Props> = ({
  uploadData,
  trainingResults,
  onSelectSheet,
  onTrainAll,
  loadingTrainAll
}) => {
  const sheets = uploadData.sheets || [];
  const trainedSheetNames = Object.keys(trainingResults);
  const totalSheets = sheets.length;
  const trainedCount = trainedSheetNames.length;

  // Compute average R2 across all trained sheets
  let totalR2 = 0;
  let highestR2 = -1;
  let highestSheet = '';

  trainedSheetNames.forEach(sName => {
    const res = trainingResults[sName];
    if (res && res.metrics) {
      const jvR2 = res.metrics.jv?.r2 || 0;
      const qeR2 = res.metrics.qe?.r2 || 0;
      const avgSheetR2 = (jvR2 + qeR2) / (jvR2 && qeR2 ? 2 : 1);
      totalR2 += avgSheetR2;
      if (avgSheetR2 > highestR2) {
        highestR2 = avgSheetR2;
        highestSheet = sName;
      }
    }
  });

  const avgR2 = trainedCount > 0 ? (totalR2 / trainedCount).toFixed(4) : 'N/A';

  return (
    <div className="animate-fade-in" style={{ width: '100%' }}>
      {/* Overview Header Banner */}
      <div className="ui-card" style={{ padding: '2rem', marginBottom: '2rem', backgroundColor: '#FFFFFF' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--color-accent)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.3rem' }}>
              Consolidated Workbook Analytics
            </div>
            <h2 className="text-card-title" style={{ fontSize: '1.6rem', marginBottom: '0.4rem' }}>
              Multi-Sheet Simulation Summary Dashboard
            </h2>
            <p className="text-muted" style={{ margin: 0 }}>
              Comparative overview of physical parameter sweeps and surrogate model accuracies across all workbook worksheets.
            </p>
          </div>

          {trainedCount < totalSheets && (
            <button className="btn-primary" onClick={onTrainAll} disabled={loadingTrainAll} style={{ padding: '0.85rem 1.5rem', fontSize: '0.95rem' }}>
              <BarChart2 size={18} />
              <span>{loadingTrainAll ? 'Batch Fitting All Worksheets...' : `Train All ${totalSheets} Worksheets`}</span>
            </button>
          )}
        </div>

        {/* 4 Overview Metric Stat Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem' }}>
          <div style={{ padding: '1.25rem', backgroundColor: 'var(--color-bg)', borderRadius: '10px', border: '1px solid var(--color-border)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-secondary)', fontWeight: 600, marginBottom: '0.2rem' }}>
              Total Worksheets
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary)' }}>
              {totalSheets}
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: 'var(--color-bg)', borderRadius: '10px', border: '1px solid var(--color-border)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-secondary)', fontWeight: 600, marginBottom: '0.2rem' }}>
              Surrogate Models Fitted
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: trainedCount === totalSheets ? 'var(--color-success)' : 'var(--color-primary)' }}>
              {trainedCount} / {totalSheets}
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: 'var(--color-bg)', borderRadius: '10px', border: '1px solid var(--color-border)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-secondary)', fontWeight: 600, marginBottom: '0.2rem' }}>
              Average Model R² Score
            </div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary)' }}>
              {avgR2 !== 'N/A' ? avgR2 : '—'}
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: 'var(--color-bg)', borderRadius: '10px', border: '1px solid var(--color-border)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--color-secondary)', fontWeight: 600, marginBottom: '0.2rem' }}>
              Highest Accuracy Sheet
            </div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--color-accent)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {highestSheet ? `${highestSheet} (${(highestR2 * 100).toFixed(1)}%)` : '—'}
            </div>
          </div>
        </div>
      </div>

      {/* Comparative Results Table */}
      <div className="ui-card" style={{ padding: '1.75rem', backgroundColor: '#FFFFFF' }}>
        <h3 className="text-card-title" style={{ fontSize: '1.2rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={20} color="var(--color-accent)" />
          <span>Worksheet Comparative Results Table</span>
        </h3>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--color-bg)', borderBottom: '2px solid var(--color-border)' }}>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>Worksheet</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>Swept Parameter</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>Curve Types</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>J-V R² Score</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>QE R² Score</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700 }}>Status</th>
                <th style={{ padding: '0.9rem 1rem', color: 'var(--color-primary)', fontWeight: 700, textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {sheets.map(s => {
                const isTrained = !!trainingResults[s.sheet_name];
                const res = trainingResults[s.sheet_name];
                const jvR2 = res?.metrics?.jv?.r2;
                const qeR2 = res?.metrics?.qe?.r2;

                return (
                  <tr key={s.sheet_name} style={{ borderBottom: '1px solid var(--color-border)', transition: 'background-color 150ms ease' }}>
                    <td style={{ padding: '1rem', fontWeight: 700, color: 'var(--color-primary)' }}>
                      {s.sheet_name}
                    </td>
                    <td style={{ padding: '1rem', color: 'var(--color-secondary)' }}>
                      <strong>{s.detected_parameter_name}</strong> ({s.detected_parameter_unit || 'Unitless'})
                    </td>
                    <td style={{ padding: '1rem' }}>
                      <div style={{ display: 'flex', gap: '0.35rem' }}>
                        {s.curve_types_found.map(ct => (
                          <span key={ct} style={{ fontSize: '0.72rem', backgroundColor: 'var(--color-bg)', border: '1px solid var(--color-border)', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: 600, fontFamily: 'monospace' }}>
                            {ct.toUpperCase()}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td style={{ padding: '1rem', fontWeight: 700, color: jvR2 && jvR2 > 0.95 ? 'var(--color-success)' : 'var(--color-primary)' }}>
                      {jvR2 !== undefined ? jvR2.toFixed(4) : '—'}
                    </td>
                    <td style={{ padding: '1rem', fontWeight: 700, color: qeR2 && qeR2 > 0.95 ? 'var(--color-success)' : 'var(--color-primary)' }}>
                      {qeR2 !== undefined ? qeR2.toFixed(4) : '—'}
                    </td>
                    <td style={{ padding: '1rem' }}>
                      {isTrained ? (
                        <span style={{ fontSize: '0.78rem', backgroundColor: '#F0FDF4', color: 'var(--color-success)', border: '1px solid #BBF7D0', padding: '0.25rem 0.6rem', borderRadius: '6px', fontWeight: 700, display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                          <CheckCircle2 size={13} />
                          <span>Trained</span>
                        </span>
                      ) : (
                        <span style={{ fontSize: '0.78rem', color: 'var(--color-secondary)', backgroundColor: 'var(--color-bg)', border: '1px solid var(--color-border)', padding: '0.25rem 0.6rem', borderRadius: '6px' }}>
                          Ready
                        </span>
                      )}
                    </td>
                    <td style={{ padding: '1rem', textAlign: 'right' }}>
                      <button
                        className="btn-secondary"
                        onClick={() => onSelectSheet(s.sheet_name)}
                        style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem' }}
                      >
                        <span>View Analytics</span>
                        <ArrowRight size={14} />
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
