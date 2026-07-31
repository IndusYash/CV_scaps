import React from 'react';
import Plot from 'react-plotly.js';
import { Activity } from 'lucide-react';

interface Props {
  curvesData: Record<string, any[]> | null;
  parameterName: string;
}

export const CurveVisualizer: React.FC<Props> = ({ curvesData, parameterName }) => {
  if (!curvesData || Object.keys(curvesData).length === 0) return null;

  const renderPlot = (title: string, xCol: string, yCol: string, dataList: any[], xTitle: string, yTitle: string) => {
    if (!dataList || dataList.length === 0) return null;

    const groups: Record<string, any[]> = {};
    const hasParam = parameterName && dataList[0][parameterName] !== undefined;

    dataList.forEach(row => {
      const pVal = hasParam ? row[parameterName] : 'Single Run';
      if (!groups[pVal]) groups[pVal] = [];
      groups[pVal].push(row);
    });

    const traces = Object.keys(groups).sort((a, b) => parseFloat(a) - parseFloat(b)).map(pVal => {
      const rows = groups[pVal].sort((a, b) => a[xCol] - b[xCol]);
      return {
        x: rows.map(r => r[xCol]),
        y: rows.map(r => r[yCol]),
        mode: 'lines' as const,
        name: hasParam ? `${parameterName}: ${pVal}` : title,
        line: { width: 2.5 }
      };
    });

    return (
      <div className="ui-card" style={{ marginBottom: '2rem', padding: '1.5rem', overflow: 'hidden', backgroundColor: '#FFFFFF' }}>
        <Plot
          data={traces}
          layout={{
            title: {
              text: title,
              font: { color: '#0F172A', family: 'Inter, sans-serif', size: 16, weight: 700 }
            },
            paper_bgcolor: '#FFFFFF',
            plot_bgcolor: '#FFFFFF',
            font: { color: '#334155', family: 'Inter, sans-serif', size: 12 },
            xaxis: {
              title: { text: xTitle, font: { color: '#0F172A', size: 13, weight: 600 } },
              tickfont: { color: '#334155', size: 11 },
              gridcolor: '#E2E8F0',
              linecolor: '#CBD5E1',
              zerolinecolor: '#94A3B8'
            },
            yaxis: {
              title: { text: yTitle, font: { color: '#0F172A', size: 13, weight: 600 } },
              tickfont: { color: '#334155', size: 11 },
              gridcolor: '#E2E8F0',
              linecolor: '#CBD5E1',
              zerolinecolor: '#94A3B8'
            },
            legend: {
              font: { color: '#0F172A', size: 12 },
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

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <h3 className="text-section-title" style={{ fontSize: '1.35rem', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem', borderLeft: '4px solid var(--color-accent)', paddingLeft: '0.8rem' }}>
        <Activity color="var(--color-accent)" size={22} />
        Spectral Characteristic Curves
      </h3>

      {curvesData.jv && renderPlot('J-V Characteristic Curves', 'Voltage', 'Current', curvesData.jv, 'Voltage (V)', 'Current Density (mA/cm²)')}
      {curvesData.qe && renderPlot('Quantum Efficiency (QE) vs Wavelength', 'Wavelength', 'QE', curvesData.qe, 'Wavelength (nm)', 'Quantum Efficiency (%)')}
      {curvesData.responsivity && renderPlot('Responsivity vs Wavelength', 'Wavelength', 'Responsivity', curvesData.responsivity, 'Wavelength (nm)', 'Responsivity (A/W)')}
      {curvesData.detectivity && renderPlot('Detectivity (D*) vs Voltage', 'Voltage', 'Detectivity', curvesData.detectivity, 'Voltage (V)', 'Detectivity D* (Jones)')}
    </div>
  );
};
