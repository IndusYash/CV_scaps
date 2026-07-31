import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

interface Props {
  sheetName: string;
  parameterName: string;
}

export const PhysicsPanel: React.FC<Props> = ({ sheetName = '', parameterName = '' }) => {
  const [open, setOpen] = useState(false);

  const getPhysicsTheory = (): string[] => {
    const sLower = (sheetName || '').toLowerCase();
    if (sLower.includes('pvk') || sLower.includes('absorber')) {
      return [
        'The Perovskite (PVK) layer functions as the primary optical absorber and photocarrier generator.',
        'Increasing absorber thickness improves total photon harvesting, especially for long wavelengths.',
        'Excessive thickness can increase bulk recombination and charge carrier transport resistance.',
        'J-V, QE, and Responsivity curves strongly depend on absorber optimization.'
      ];
    } else if (sLower.includes('etl') || sLower.includes('electron')) {
      return [
        'The Electron Transport Layer (ETL) selectively extracts electrons and blocks holes.',
        'ETL thickness impacts series resistance, optical transmittance, and band alignment.',
        'Optimal ETL thickness minimizes interfacial recombination while maintaining high conductivity.'
      ];
    } else if (sLower.includes('htl') || sLower.includes('hole')) {
      return [
        'The Hole Transport Layer (HTL) extracts holes to the anode while blocking electrons.',
        'HTL thickness influences hole mobility, series resistance, and interfacial energy level matching.'
      ];
    } else if (sLower.includes('temp')) {
      return [
        'Temperature affects intrinsic carrier concentration, semiconductor bandgap, and carrier mobility.',
        'Higher temperatures increase dark saturation current, reducing open-circuit voltage (Voc).'
      ];
    } else if (sLower.includes('defect') || sLower.includes('interface')) {
      return [
        'Defect and interface state densities act as Shockley-Read-Hall (SRH) recombination centers.',
        'Higher defect densities reduce carrier lifetime, Voc, and fill factor (FF).'
      ];
    } else {
      return [
        'This sheet analyzes physical parameter sweeps across solar cell operational layers.',
        'Machine learning models capture nonlinear relationships across J-V, QE, and spectral metrics.'
      ];
    }
  };

  const points = getPhysicsTheory();

  return (
    <div className="glass-card" style={{ marginBottom: '2rem' }}>
      <div
        onClick={() => setOpen(!open)}
        style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <BookOpen color="var(--accent-gold)" size={22} />
          <h3>Physical Interpretation & Photovoltaic Theory — {sheetName || 'Worksheet'}</h3>
        </div>
        <button className="btn-secondary" style={{ padding: '0.3rem 0.6rem' }}>
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </button>
      </div>

      {open && (
        <div style={{ marginTop: '1rem', borderTop: '1px solid var(--color-border)', paddingTop: '1rem' }}>
          <ul style={{ paddingLeft: '1.2rem', margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {points.map((pt, idx) => (
              <li key={idx}>{pt}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
