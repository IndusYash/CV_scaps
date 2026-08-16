import React from 'react';
import Plot from 'react-plotly.js';
import { Award } from 'lucide-react';

interface Props {
  mlResults: any | null;
}

export const ParityPlots: React.FC<Props> = ({ mlResults }) => {
  if (!mlResults) {
    return (
      <div className="ui-card" style={{ padding: '2rem', textAlign: 'center', marginTop: '2rem' }}>
        <p className="text-body">No Random Forest regression models trained yet. Train the V1 model on the database JV parameters using the controls above.</p>
      </div>
    );
  }

  if (mlResults.status === 'INSUFFICIENT_DATA') {
    return (
      <div className="ui-card" style={{ padding: '2rem', textAlign: 'center', marginTop: '2rem', borderLeft: '4px solid red' }}>
        <h4 style={{ color: 'red', margin: '0 0 0.5rem 0' }}>Insufficient Data for Global Model Training</h4>
        <p className="text-body">The database does not contain enough datasets/traces to perform validation splitting. Please upload more SCAPS Excel workbooks to meet the model training requirements.</p>
      </div>
    );
  }

  if (mlResults.status !== 'SUCCESS') {
    return (
      <div className="ui-card" style={{ padding: '2rem', textAlign: 'center', marginTop: '2rem' }}>
        <p className="text-body">No Random Forest regression models trained yet. Train the V1 model on the database JV parameters using the controls above.</p>
      </div>
    );
  }

  const { metrics, targets } = mlResults;

  if (!metrics || !targets) {
    return (
      <div className="ui-card" style={{ padding: '2rem', textAlign: 'center', marginTop: '2rem' }}>
        <p className="text-body" style={{ color: 'red' }}>Error: Prediction targets payload is unavailable in the ML results.</p>
      </div>
    );
  }

  const renderPlot = (targetKey: string, titleLabel: string) => {
    const targetData = targets[targetKey];
    if (!targetData) {
      return (
        <div className="ui-card" style={{ padding: '1rem', color: 'red' }}>
          Error: Predictions unavailable for '{targetKey}'.
        </div>
      );
    }

    const actuals = targetData.actual || [];
    const predicted = targetData.predicted || [];
    const minVal = Math.min(...actuals, ...predicted, 0.0);
    const maxVal = Math.max(...actuals, ...predicted) * 1.05;

    return (
      <div className="ui-card animate-fade-in" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', height: '360px' }}>
        <Plot
          data={[
            {
              x: [minVal, maxVal],
              y: [minVal, maxVal],
              type: 'scatter',
              mode: 'lines',
              line: { color: 'rgba(15, 23, 42, 0.4)', width: 1.5, dash: 'dash' },
              name: 'y = x Reference',
              showlegend: false,
            },
            {
              x: actuals,
              y: predicted,
              type: 'scatter',
              mode: 'markers',
              marker: {
                color: 'var(--color-success)',
                size: 7,
                opacity: 0.8,
                line: { color: '#ffffff', width: 0.75 }
              },
              hovertemplate: '<b>Actual</b>: %{x:.4f}<br><b>Predicted</b>: %{y:.4f}<extra></extra>',
              name: 'Random Forest',
              showlegend: false,
            }
          ]}
          layout={{
            autosize: true,
            margin: { l: 50, r: 15, t: 35, b: 40 },
            title: {
              text: titleLabel,
              font: { family: 'Inter', size: 12, weight: 600 }
            },
            xaxis: {
              title: 'Actual',
              gridcolor: 'var(--color-border)',
              zeroline: false,
              range: [minVal, maxVal],
            },
            yaxis: {
              title: 'Predicted',
              gridcolor: 'var(--color-border)',
              zeroline: false,
              range: [minVal, maxVal],
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{ displayModeBar: false }}
        />
        <div style={{ marginTop: '0.4rem', fontSize: '0.75rem', color: 'var(--color-secondary)', display: 'flex', gap: '0.8rem', justifyContent: 'center' }}>
          <span>R²: <strong>{targetData.metrics.r2.toFixed(4)}</strong></span>
          <span>MAE: <strong>{targetData.metrics.mae.toFixed(4)}</strong></span>
          <span>RMSE: <strong>{targetData.metrics.rmse.toFixed(4)}</strong></span>
          <span>Samples: <strong>{targetData.metrics.n_samples}</strong></span>
        </div>
      </div>
    );
  };

  return (
    <div className="animate-fade-in" style={{ marginTop: '2rem' }}>
      <h3 className="text-section-title" style={{ fontSize: '1.35rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '4px solid var(--color-success)', paddingLeft: '0.8rem' }}>
        <Award color="var(--color-success)" size={22} />
        Random Forest PV Parameter Parity Metrics (V1)
      </h3>

      {/* Validation configuration details card */}
      <div className="ui-card animate-fade-in" style={{ padding: '1.2rem', marginBottom: '1.5rem', borderLeft: '4px solid var(--color-accent)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--color-secondary)', fontWeight: 700 }}>Active Validation Mode</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
              <span style={{ 
                fontSize: '0.82rem', 
                fontWeight: 700, 
                backgroundColor: mlResults.validation_mode === 'unseen_workbook' ? 'var(--color-accent-light)' : 'rgba(34, 197, 94, 0.1)', 
                color: mlResults.validation_mode === 'unseen_workbook' ? 'var(--color-accent)' : 'var(--color-success)', 
                padding: '0.2rem 0.6rem', 
                borderRadius: '6px' 
              }}>
                {mlResults.validation_mode === 'unseen_workbook' ? 'True Unseen Workbook' : 'Within-Dataset Interpolation'}
              </span>
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--color-secondary)', fontWeight: 700 }}>Training Set Workbooks</div>
            <div style={{ fontSize: '0.84rem', color: 'var(--color-primary)', fontWeight: 600, marginTop: '0.2rem', maxWidth: '300px', wordBreak: 'break-all' }}>
              {mlResults.train_workbooks?.join(', ') || 'N/A'}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--color-secondary)', fontWeight: 700 }}>Test / Validation Set</div>
            <div style={{ fontSize: '0.84rem', color: 'var(--color-primary)', fontWeight: 600, marginTop: '0.2rem', maxWidth: '300px', wordBreak: 'break-all' }}>
              {mlResults.validation_mode === 'unseen_workbook' 
                ? `Held Out Workbook: ${mlResults.test_workbooks?.join(', ') || mlResults.test_workbook || 'N/A'}`
                : `Interior sweeps (${(targets.pce?.metrics?.n_samples || 0)} test samples)`}
            </div>
          </div>
        </div>

        {/* Warning if sample size is insufficient */}
        {mlResults.is_statistically_meaningful === false && (
          <div style={{ 
            marginTop: '1rem', 
            padding: '0.8rem 1rem', 
            backgroundColor: '#FEF3C7', 
            color: '#92400E', 
            borderRadius: '6px', 
            fontSize: '0.82rem', 
            fontWeight: 600,
            borderLeft: '4px solid #D97706'
          }}>
            ⚠️ Warning: Test set has insufficient samples ({targets.pce?.metrics?.n_samples || 0}). A minimum of 5 test samples is required for statistically meaningful validation. Reported metrics represent individual sample validation errors rather than generalized performance.
          </div>
        )}
      </div>

      {/* 2x2 Grid of parity plots */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {renderPlot('j_sc', 'Jsc Validation (Short-Circuit Current Density, mA/cm²)')}
        {renderPlot('v_oc', 'Voc Validation (Open-Circuit Voltage, V)')}
        {renderPlot('ff', 'FF Validation (Fill Factor)')}
        {renderPlot('pce', 'PCE Validation (Power Conversion Efficiency, %)')}
      </div>

      {/* Summary metrics table */}
      <div className="ui-card animate-fade-in" style={{ padding: '1.5rem' }}>
        <h4 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: 'var(--color-primary)' }}>Performance Summary Metrics</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--color-border)', textAlign: 'left' }}>
              <th style={{ padding: '0.5rem 1rem 0.5rem 0', color: 'var(--color-secondary)' }}>Target</th>
              <th style={{ padding: '0.5rem 1rem', color: 'var(--color-secondary)' }}>MAE</th>
              <th style={{ padding: '0.5rem 1rem', color: 'var(--color-secondary)' }}>RMSE</th>
              <th style={{ padding: '0.5rem 1rem', color: 'var(--color-secondary)' }}>R²</th>
              <th style={{ padding: '0.5rem 0', color: 'var(--color-secondary)' }}>Test Samples</th>
            </tr>
          </thead>
          <tbody>
            {['j_sc', 'v_oc', 'ff', 'pce'].map((tKey) => {
              const targetData = targets[tKey];
              if (!targetData) return null;

              const nameMap: Record<string, string> = {
                j_sc: 'Jsc (Short-Circuit Current Density, mA/cm²)',
                v_oc: 'Voc (Open-Circuit Voltage, V)',
                ff: 'FF (Fill Factor)',
                pce: 'PCE (Power Conversion Efficiency, %)',
              };

              return (
                <tr key={tKey} style={{ borderBottom: '1px solid var(--color-border)' }}>
                  <td style={{ padding: '0.75rem 1rem 0.75rem 0', fontWeight: 600, color: 'var(--color-primary)' }}>{nameMap[tKey]}</td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace' }}>{targetData.metrics.mae.toFixed(5)}</td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace' }}>{targetData.metrics.rmse.toFixed(5)}</td>
                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', fontWeight: 700, color: targetData.metrics.r2 > 0.8 ? 'var(--color-success)' : 'inherit' }}>{targetData.metrics.r2.toFixed(5)}</td>
                  <td style={{ padding: '0.75rem 0' }}>{targetData.metrics.n_samples}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
