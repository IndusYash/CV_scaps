import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer style={{ backgroundColor: '#FFFFFF', borderTop: '1px solid var(--color-border)', padding: '3.5rem 0 2.5rem 0', width: '100%' }}>
      <div className="max-content">
        <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr 1fr', gap: '3rem', paddingBottom: '2.5rem', borderBottom: '1px solid var(--color-border)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '0.8rem' }}>
              <img src="/mmmut_logo.png" alt="MMMUT Logo" style={{ height: '36px', width: 'auto' }} />
              <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--color-primary)' }}>
                Computer Vision Laboratory
              </span>
            </div>
            <p style={{ fontSize: '0.88rem', color: 'var(--color-secondary)', lineHeight: 1.6 }}>
              Department of Computer Science & Engineering<br />
              Madan Mohan Malaviya University of Technology (MMMUT), Gorakhpur
            </p>
          </div>

          <div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Research Collaborations
            </div>
            <p style={{ fontSize: '0.88rem', color: 'var(--color-secondary)', lineHeight: 1.6 }}>
              Built for <strong>Photonics Research Laboratory</strong><br />
              Department of Physics and Material Science, MMMUT
            </p>
          </div>

          <div>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--color-primary)', marginBottom: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Faculty Supervision
            </div>
            <p style={{ fontSize: '0.88rem', color: 'var(--color-secondary)', lineHeight: 1.6 }}>
              <strong>Dr. Satvik Vats</strong><br />
              Assistant Professor<br />
              Department of CSE, MMMUT
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '2rem', fontSize: '0.84rem', color: 'var(--color-secondary)' }}>
          <span>© 2026 MMMUT Computer Vision Laboratory. All rights reserved.</span>
          <span>Version 1.0 • Platform Release</span>
        </div>
      </div>
    </footer>
  );
};
