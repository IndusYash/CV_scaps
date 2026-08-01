import React, { useState, useMemo, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Activity, Filter, AlertTriangle, Bug } from 'lucide-react';
import { ExperimentPayload, TracePayload } from '../api/client';

interface Props {
  experiments: ExperimentPayload[] | null;
  curvesData?: Record<string, any[]> | null;
  parameterName?: string;
}

export const CurveVisualizer: React.FC<Props> = ({ experiments }) => {
  if (!experiments || experiments.length === 0) return null;

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <h3 className="text-section-title" style={{ fontSize: '1.35rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '4px solid var(--color-accent)', paddingLeft: '0.8rem' }}>
        <Activity color="var(--color-accent)" size={22} />
        Scientific Experiment Visualizations
      </h3>

      {experiments.map((exp, idx) => (
        <ExperimentCard key={`${exp.experiment_id}_${exp.target.name}`} experiment={exp} index={idx} />
      ))}
    </div>
  );
};

interface CardProps {
  experiment: ExperimentPayload;
  index: number;
}

const ExperimentCard: React.FC<CardProps> = ({ experiment, index }) => {
  const { experiment_id, sheet_name, semantic_type, human_label, axis, target, parameters, traces } = experiment;

  // Track parameter filter states for multi-parameter experiments
  const [selectedParams, setSelectedParams] = useState<Record<string, number[]>>(() => {
    const initial: Record<string, number[]> = {};
    parameters.forEach(p => {
      if (p.values.length > 0) {
        if (p.values.length > 5) {
          initial[p.name] = p.values.slice(0, 4);
        } else {
          initial[p.name] = p.values;
        }
      }
    });
    return initial;
  });

  // Filter traces based on selected parameters
  const visibleTraces = useMemo(() => {
    if (parameters.length === 0) return traces;

    return traces.filter(tr => {
      return Object.entries(selectedParams).every(([pName, allowedVals]) => {
        if (!allowedVals || allowedVals.length === 0) return true;
        const trVal = tr.parameters[pName];
        if (trVal === undefined) return true;
        return allowedVals.includes(trVal);
      });
    });
  }, [traces, selectedParams, parameters]);

  // Log filter changes in development
  useEffect(() => {
    if (process.env.NODE_ENV !== 'production') {
      const matchedIds = visibleTraces.map(t => t.trace_id);
      console.log(`[SCAPS FILTER] ${experiment_id}`, {
        selectedParams,
        matchingTraceIds: matchedIds,
        resultingPlotlyTraceCount: visibleTraces.length
      });
    }
  }, [selectedParams, visibleTraces, experiment_id]);

  // Comprehensive Sanity Checks, Extreme Value Warnings, and Graph Debug Logging
  const { plotlyTraces, sanityWarnings, scientificWarnings } = useMemo(() => {
    const pTraces: any[] = [];
    const sWarnings: string[] = [];
    const scWarnings: string[] = [];

    visibleTraces.forEach((tr, tIdx) => {
      const traceKey = tr.trace_id || `trace_${tIdx + 1}`;
      
      // 1. Plotly Invariant Sanity Checks
      if (tr.x.length !== tr.y.length) {
        const warnMsg = `[SCAPS PLOTLY INVARIANT WARNING] ${sheet_name} | Exp: ${experiment_id} | Trace: ${traceKey} | Block: ${tr.source_block_id} | Failed: x.length (${tr.x.length}) !== y.length (${tr.y.length})`;
        console.warn(warnMsg);
        sWarnings.push(warnMsg);
      }

      if (tr.x.length < 2) {
        const warnMsg = `[SCAPS PLOTLY INVARIANT WARNING] ${sheet_name} | Exp: ${experiment_id} | Trace: ${traceKey} | Block: ${tr.source_block_id} | Failed: points count (${tr.x.length}) < 2`;
        console.warn(warnMsg);
        sWarnings.push(warnMsg);
      }

      const hasInvalidVal = tr.x.some(v => isNaN(v) || !isFinite(v) || v === null || v === undefined) ||
                             tr.y.some(v => isNaN(v) || !isFinite(v) || v === null || v === undefined);
      if (hasInvalidVal) {
        const warnMsg = `[SCAPS PLOTLY INVARIANT WARNING] ${sheet_name} | Exp: ${experiment_id} | Trace: ${traceKey} | Block: ${tr.source_block_id} | Failed: contains NaN, Infinity, or undefined numerical values`;
        console.warn(warnMsg);
        sWarnings.push(warnMsg);
      }

      // Check Monotonic X ordering
      let isMonotonic = true;
      for (let i = 1; i < tr.x.length; i++) {
        if (tr.x[i] < tr.x[i - 1]) {
          isMonotonic = false;
          break;
        }
      }
      if (!isMonotonic && tr.x.length > 2) {
        console.info(`[SCAPS X ORDERING INFO] ${sheet_name} | Exp: ${experiment_id} | Trace: ${traceKey} | X-axis is non-monotonic or reverse-ordered`);
      }

      // 2. Extreme Value Scientific Warnings (e.g. QE < 0% or > 100%)
      if (target.name === 'QE') {
        tr.y.forEach((val, pIdx) => {
          if (val < 0.0 || val > 100.0) {
            const scMsg = `[SCAPS SCIENTIFIC WARNING] Exp: ${experiment_id} | Trace: ${traceKey} | Target: ${target.name} | Value: ${val} % at X=${tr.x[pIdx]} ${axis.unit} | Block: ${tr.source_block_id}`;
            console.warn(scMsg);
            if (scWarnings.length < 5) scWarnings.push(scMsg);
          }
        });
      }

      pTraces.push({
        x: tr.x,
        y: tr.y,
        mode: 'lines' as const,
        name: tr.label || traceKey,
        line: { width: 2.5 }
      });
    });

    // 3. Overall Development Console Diagnostic Log
    if (process.env.NODE_ENV !== 'production') {
      console.group(`[SCAPS GRAPH DEBUG] Sheet: ${sheet_name} | Exp: ${experiment_id}`);
      console.log(`Semantic Type: ${semantic_type} | Target: ${target.name} (${target.unit}) | Axis: ${axis.name} (${axis.unit})`);
      console.log(`Available Traces: ${traces.length} | Visible Traces: ${visibleTraces.length}`);
      
      visibleTraces.forEach(tr => {
        const xMin = tr.x.length ? Math.min(...tr.x) : 0;
        const xMax = tr.x.length ? Math.max(...tr.x) : 0;
        const yMin = tr.y.length ? Math.min(...tr.y) : 0;
        const yMax = tr.y.length ? Math.max(...tr.y) : 0;
        console.log(`  Trace: ${tr.trace_id} | Label: '${tr.label}' | Block: ${tr.source_block_id} | Points: ${tr.x.length} | X: ${xMin} -> ${xMax} | Y: ${yMin} -> ${yMax} | Params:`, tr.parameters);
      });
      console.groupEnd();
    }

    return { plotlyTraces: pTraces, sanityWarnings: sWarnings, scientificWarnings: scWarnings };
  }, [visibleTraces, sheet_name, experiment_id, semantic_type, target, axis, traces]);

  const handleSingleParamChange = (paramName: string, value: number) => {
    setSelectedParams(prev => ({
      ...prev,
      [paramName]: [value]
    }));
  };

  const handleToggleMultiParam = (paramName: string, value: number) => {
    setSelectedParams(prev => {
      const current = prev[paramName] || [];
      const updated = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value];
      return { ...prev, [paramName]: updated };
    });
  };

  return (
    <div className="ui-card" style={{ marginBottom: '2rem', padding: '1.5rem', overflow: 'hidden', backgroundColor: '#FFFFFF' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #E2E8F0', paddingBottom: '0.75rem' }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '1.1rem', color: '#0F172A', fontWeight: 700 }}>
            {human_label}
          </h4>
          <span style={{ fontSize: '0.8rem', color: '#64748B' }}>
            ID: {experiment_id} | Axis: {axis.name} ({axis.unit}) vs Target: {target.name} ({target.unit}) | Available Traces: {traces.length}
          </span>
        </div>
        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#2563EB', backgroundColor: '#EFF6FF', padding: '0.3rem 0.75rem', borderRadius: '6px' }}>
          {visibleTraces.length} / {traces.length} Traces Displayed
        </span>
      </div>

      {/* Parameter Filter Controls Banner */}
      {parameters.length > 0 && (
        <div style={{ backgroundColor: '#F8FAFC', padding: '0.9rem', borderRadius: '8px', marginBottom: '1rem', border: '1px solid #E2E8F0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', fontWeight: 600, color: '#334155', marginBottom: '0.6rem' }}>
            <Filter size={16} color="#2563EB" />
            Parameter Sweep Display Filter Controls
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.25rem' }}>
            {parameters.map(p => {
              const currentSelected = selectedParams[p.name] || [];
              const isMultiSelect = p.values.length > 3;

              return (
                <div key={p.name} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                  <label style={{ fontSize: '0.78rem', fontWeight: 600, color: '#475569' }}>
                    {p.name} ({p.unit})
                  </label>

                  {!isMultiSelect ? (
                    <select
                      value={currentSelected[0] ?? p.values[0]}
                      onChange={e => handleSingleParamChange(p.name, parseFloat(e.target.value))}
                      style={{ padding: '0.35rem 0.6rem', fontSize: '0.82rem', borderRadius: '5px', border: '1px solid #CBD5E1', backgroundColor: '#FFFFFF' }}
                    >
                      {p.values.map(val => (
                        <option key={val} value={val}>
                          {val} {p.unit}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                      {p.values.map(val => {
                        const isChecked = currentSelected.includes(val);
                        return (
                          <button
                            key={val}
                            type="button"
                            onClick={() => handleToggleMultiParam(p.name, val)}
                            style={{
                              padding: '0.2rem 0.5rem',
                              fontSize: '0.75rem',
                              borderRadius: '4px',
                              border: isChecked ? '1px solid #2563EB' : '1px solid #CBD5E1',
                              backgroundColor: isChecked ? '#DBEAFE' : '#FFFFFF',
                              color: isChecked ? '#1E40AF' : '#475569',
                              cursor: 'pointer',
                              fontWeight: isChecked ? 600 : 400
                            }}
                          >
                            {val}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Warnings Banner (Dev mode) */}
      {(sanityWarnings.length > 0 || scientificWarnings.length > 0) && (
        <div style={{ backgroundColor: '#FEF2F2', border: '1px solid #FCA5A5', padding: '0.75rem 1rem', borderRadius: '6px', marginBottom: '1rem', fontSize: '0.82rem', color: '#991B1B' }}>
          <div style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.3rem' }}>
            <AlertTriangle size={16} color="#DC2626" /> Diagnostic Warnings Detected
          </div>
          {sanityWarnings.map((w, idx) => (
            <div key={`sw_${idx}`}>{w}</div>
          ))}
          {scientificWarnings.map((w, idx) => (
            <div key={`scw_${idx}`}>{w}</div>
          ))}
        </div>
      )}

      {/* Plotly Figure */}
      <Plot
        data={plotlyTraces}
        layout={{
          title: {
            text: `${target.name} vs ${axis.name}`,
            font: { color: '#0F172A', family: 'Inter, sans-serif', size: 16, weight: 700 }
          },
          paper_bgcolor: '#FFFFFF',
          plot_bgcolor: '#FFFFFF',
          font: { color: '#334155', family: 'Inter, sans-serif', size: 12 },
          xaxis: {
            title: { text: `${axis.name} (${axis.unit})`, font: { color: '#0F172A', size: 13, weight: 600 } },
            tickfont: { color: '#334155', size: 11 },
            gridcolor: '#E2E8F0',
            linecolor: '#CBD5E1',
            zerolinecolor: '#94A3B8'
          },
          yaxis: {
            title: { text: `${target.name} (${target.unit})`, font: { color: '#0F172A', size: 13, weight: 600 } },
            tickfont: { color: '#334155', size: 11 },
            gridcolor: '#E2E8F0',
            linecolor: '#CBD5E1',
            zerolinecolor: '#94A3B8'
          },
          legend: {
            font: { color: '#0F172A', size: 11 },
            bgcolor: 'rgba(255, 255, 255, 0.9)',
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
  );
};
