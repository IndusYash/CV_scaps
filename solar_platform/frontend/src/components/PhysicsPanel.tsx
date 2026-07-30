import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react';

interface Props {
  sheetName: string;
  parameterName: string;
}

export const PhysicsPanel: React.FC<Props> = ({ sheetName, parameterName }) => {
  const [open, setOpen] = useState(false);

  const getPhysicsTheory = (): string[] => {
    const sLower = sheetName.toLowerCase();
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
          <h3>Physical Interpretation & Photovoltaic Theory — {sheetName}</h3>
        </div>
        {open ? <ChevronUp color="var(--accent-gold)" /> : <ChevronDown color="var(--accent-gold)" />}
      </div>

      {open && (
        <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid var(--border-glow)' }}>
          <ul style={{ paddingLeft: '1.5rem', color: 'var(--text-muted)' }}>
            {points.map((pt, i) => (
              <li key={i} style={{ marginBottom: '0.5rem' }}>{pt}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
