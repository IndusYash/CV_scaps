import React, { useState, useRef } from 'react';
import {
  Upload,
  FileSpreadsheet,
  AlertCircle,
  Cpu,
  LineChart,
  Database,
  Sliders,
  CheckCircle2,
  Download
} from 'lucide-react';
import { uploadExcelFile, UploadResponse } from '../api/client';
import { Footer } from './Footer';

interface Props {
  onUploadSuccess: (data: UploadResponse) => void;
}

export const FileUploader: React.FC<Props> = ({ onUploadSuccess }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingStep, setLoadingStep] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (file: File | null) => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setLoadingStep(1);

    const t1 = setTimeout(() => setLoadingStep(2), 500);
    const t2 = setTimeout(() => setLoadingStep(3), 1000);
    const t3 = setTimeout(() => setLoadingStep(4), 1500);

    try {
      const data = await uploadExcelFile(file);
      setLoadingStep(5);
      setTimeout(() => {
        onUploadSuccess(data);
      }, 400);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to parse Excel file. Please ensure it is a valid SCAPS workbook (.xlsx).');
      setLoading(false);
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleUploadButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const scrollToFeatures = () => {
    const el = document.getElementById('features-section');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="animate-fade-in" style={{ width: '100%' }}>
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".xlsx, .xls"
        onChange={(e) => handleFileChange(e.target.files ? e.target.files[0] : null)}
        style={{ display: 'none' }}
        disabled={loading}
      />

      {/* HERO SECTION WITH DIRECT FILE INPUT */}
      <section
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        style={{
          padding: '4.5rem 0 5rem 0',
          borderBottom: '1px solid var(--color-border)',
          backgroundColor: isDragging ? 'var(--color-accent-light)' : 'transparent',
          transition: 'background-color 200ms ease'
        }}
      >
        <div className="max-content">
          {!loading ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1.15fr 0.85fr', gap: '3.5rem', alignItems: 'center', minHeight: '440px' }}>
              {/* LEFT COLUMN */}
              <div>
                <div style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  backgroundColor: 'var(--color-accent-light)',
                  border: '1px solid #BFDBFE',
                  borderRadius: '999px',
                  padding: '0.35rem 0.9rem',
                  fontSize: '0.84rem',
                  fontWeight: 600,
                  color: 'var(--color-accent)',
                  marginBottom: '1.5rem'
                }}>
                  <span>SCAPS-1D Simulation & Machine Learning Surrogate Interface</span>
                </div>

                <h1 className="text-hero-title" style={{ marginBottom: '1.25rem' }}>
                  Photovoltaic Analytics Platform
                </h1>

                <p className="text-body" style={{ fontSize: '1.125rem', marginBottom: '2.25rem', maxWidth: '620px', lineHeight: 1.6 }}>
                  Upload photovoltaic simulation datasets and generate rapid AI-assisted predictions using our Random Forest surrogate model.
                </p>

                {/* Direct Upload CTA Button */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <button className="btn-primary" onClick={handleUploadButtonClick} style={{ padding: '0.9rem 1.8rem', fontSize: '1rem' }}>
                    <Upload size={18} />
                    <span>Upload Dataset (.xlsx)</span>
                  </button>

                  <button className="btn-secondary" onClick={scrollToFeatures} style={{ padding: '0.9rem 1.6rem', fontSize: '0.95rem' }}>
                    <span>Documentation</span>
                  </button>
                </div>

                <div style={{ marginTop: '1.5rem', fontSize: '0.82rem', color: 'var(--color-secondary)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <span>Supported Formats: <strong style={{ color: 'var(--color-primary)' }}>.xlsx, .xls</strong></span>
                  <span>•</span>
                  <span>Drag & Drop files anywhere on page</span>
                </div>
              </div>

              {/* RIGHT COLUMN (Monochrome Line-Art Diagram) */}
              <div className="ui-card" style={{ padding: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#FFFFFF' }}>
                <svg width="100%" height="280" viewBox="0 0 440 280" fill="none" xmlns="http://www.w3.org/2000/svg">
                  {/* Grid Background */}
                  <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#F1F5F9" strokeWidth="1" />
                  </pattern>
                  <rect width="440" height="280" fill="url(#grid)" rx="12" />

                  {/* Solar Panel Array Nodes */}
                  <rect x="30" y="40" width="100" height="70" rx="8" stroke="#0F172A" strokeWidth="1.5" fill="#F8FAFC" />
                  <line x1="30" y1="75" x2="130" y2="75" stroke="#CBD5E1" strokeWidth="1" />
                  <line x1="80" y1="40" x2="80" y2="110" stroke="#CBD5E1" strokeWidth="1" />
                  <text x="45" y="95" fill="#64748B" fontSize="10" fontWeight="600" fontFamily="Inter">SCAPS-1D PV</text>

                  {/* Connector Line 1 */}
                  <path d="M 130 75 L 180 75" stroke="#2563EB" strokeWidth="2" strokeDasharray="4 4" />

                  {/* Feature Extraction Node */}
                  <rect x="180" y="40" width="100" height="70" rx="8" stroke="#0F172A" strokeWidth="1.5" fill="#F8FAFC" />
                  <circle cx="230" cy="65" r="14" stroke="#2563EB" strokeWidth="1.5" fill="#EFF6FF" />
                  <text x="195" y="95" fill="#64748B" fontSize="10" fontWeight="600" fontFamily="Inter">Feature Matrix</text>

                  {/* Connector Line 2 */}
                  <path d="M 280 75 L 330 75" stroke="#2563EB" strokeWidth="2" strokeDasharray="4 4" />

                  {/* ML Random Forest Surrogate Node */}
                  <rect x="330" y="40" width="80" height="70" rx="8" stroke="#0F172A" strokeWidth="1.5" fill="#0F172A" />
                  <text x="345" y="70" fill="#FFFFFF" fontSize="11" fontWeight="700" fontFamily="Inter">RF AI</text>

                  {/* Lower Spectral Curve Plot Line */}
                  <path d="M 40 210 Q 120 130 200 180 T 400 160" stroke="#2563EB" strokeWidth="2.5" fill="none" />
                  <path d="M 40 210 Q 120 150 200 195 T 400 180" stroke="#94A3B8" strokeWidth="1.5" strokeDasharray="3 3" fill="none" />

                  {/* Graph Axis */}
                  <line x1="40" y1="230" x2="400" y2="230" stroke="#CBD5E1" strokeWidth="1.5" />
                  <line x1="40" y1="130" x2="40" y2="230" stroke="#CBD5E1" strokeWidth="1.5" />
                  <text x="360" y="248" fill="#64748B" fontSize="10" fontFamily="Inter">Wavelength (nm)</text>
                  <text x="10" y="140" fill="#64748B" fontSize="10" fontFamily="Inter">J-V / QE</text>
                </svg>
              </div>
            </div>
          ) : (
            /* INLINE PROGRESS LOADING INTERFACE */
            <div className="ui-card" style={{ padding: '3.5rem 2.5rem', textAlign: 'center', maxWidth: '680px', margin: '0 auto' }}>
              <div style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                border: '3px solid #E2E8F0',
                borderTopColor: 'var(--color-accent)',
                animation: 'spin 1s linear infinite',
                margin: '0 auto 1.75rem auto'
              }} />

              <h3 className="text-card-title" style={{ marginBottom: '1.5rem' }}>
                Processing Dataset & Training Surrogate Models
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', textAlign: 'left', maxWidth: '420px', margin: '0 auto' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.92rem', color: loadingStep >= 1 ? 'var(--color-primary)' : 'var(--color-secondary)' }}>
                  <CheckCircle2 size={18} color={loadingStep >= 1 ? 'var(--color-success)' : '#CBD5E1'} />
                  <span>Uploading Dataset Workbook</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.92rem', color: loadingStep >= 2 ? 'var(--color-primary)' : 'var(--color-secondary)' }}>
                  <CheckCircle2 size={18} color={loadingStep >= 2 ? 'var(--color-success)' : '#CBD5E1'} />
                  <span>Parsing Excel Sheets & Curve Schemas</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.92rem', color: loadingStep >= 3 ? 'var(--color-primary)' : 'var(--color-secondary)' }}>
                  <CheckCircle2 size={18} color={loadingStep >= 3 ? 'var(--color-success)' : '#CBD5E1'} />
                  <span>Extracting Physical Feature Matrices</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.92rem', color: loadingStep >= 4 ? 'var(--color-primary)' : 'var(--color-secondary)' }}>
                  <CheckCircle2 size={18} color={loadingStep >= 4 ? 'var(--color-success)' : '#CBD5E1'} />
                  <span>Initializing Random Forest Regressor</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.92rem', color: loadingStep >= 5 ? 'var(--color-primary)' : 'var(--color-secondary)' }}>
                  <CheckCircle2 size={18} color={loadingStep >= 5 ? 'var(--color-success)' : '#CBD5E1'} />
                  <span>Generating Visualizations & Curves</span>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div style={{ marginTop: '1.5rem', color: '#B91C1C', backgroundColor: '#FEF2F2', border: '1px solid #FECACA', padding: '1rem', borderRadius: '12px', fontSize: '0.9rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem', maxWidth: '680px', margin: '1.5rem auto 0 auto' }}>
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </section>

      {/* PLATFORM FEATURES SECTION */}
      <section id="features-section" style={{ padding: '5rem 0', borderBottom: '1px solid var(--color-border)' }}>
        <div className="max-content">
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <h2 className="text-section-title" style={{ marginBottom: '0.5rem' }}>
              Platform Capabilities
            </h2>
            <p className="text-body" style={{ maxWidth: '600px', margin: '0 auto' }}>
              Engineered specifically for photovoltaic research scholars and device physicists.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
            {/* Feature Card 1 */}
            <div className="ui-card ui-card-hover" style={{ padding: '2rem' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'var(--color-accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
                <FileSpreadsheet size={24} color="var(--color-accent)" />
              </div>
              <h3 className="text-card-title" style={{ marginBottom: '0.5rem' }}>
                Automatic Dataset Parsing
              </h3>
              <p className="text-muted" style={{ fontSize: '0.95rem' }}>
                Automatically detects JV, QE, Spectral Responsivity, and parameter sweep columns across multi-worksheet Excel files.
              </p>
            </div>

            {/* Feature Card 2 */}
            <div className="ui-card ui-card-hover" style={{ padding: '2rem' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'var(--color-accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
                <Cpu size={24} color="var(--color-accent)" />
              </div>
              <h3 className="text-card-title" style={{ marginBottom: '0.5rem' }}>
                Machine Learning Prediction
              </h3>
              <p className="text-muted" style={{ fontSize: '0.95rem' }}>
                Trains Scikit-Learn Random Forest surrogate models for rapid, sub-second physical curve predictions.
              </p>
            </div>

            {/* Feature Card 3 */}
            <div className="ui-card ui-card-hover" style={{ padding: '2rem' }}>
              <div style={{ width: '48px', height: '48px', borderRadius: '12px', backgroundColor: 'var(--color-accent-light)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem' }}>
                <LineChart size={24} color="var(--color-accent)" />
              </div>
              <h3 className="text-card-title" style={{ marginBottom: '0.5rem' }}>
                Scientific Visualization
              </h3>
              <p className="text-muted" style={{ fontSize: '0.95rem' }}>
                Generates publication-quality interactive Plotly figures with high-resolution vector export capabilities.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* WORKFLOW TIMELINE SECTION */}
      <section style={{ padding: '5rem 0', borderBottom: '1px solid var(--color-border)', backgroundColor: '#FFFFFF' }}>
        <div className="max-content">
          <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
            <h2 className="text-section-title" style={{ marginBottom: '0.5rem' }}>
              Surrogate Pipeline Workflow
            </h2>
            <p className="text-body" style={{ maxWidth: '600px', margin: '0 auto' }}>
              From raw SCAPS-1D output files to interactive machine learning predictions in six automated steps.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '1rem', textAlign: 'center' }}>
            {[
              { step: '01', title: 'Upload', desc: 'SCAPS Excel Dataset', icon: Upload },
              { step: '02', title: 'Data Processing', desc: 'Schema Standardization', icon: Database },
              { step: '03', title: 'Feature Extraction', desc: 'Sweeps & Targets', icon: Sliders },
              { step: '04', title: 'RF Prediction', desc: 'Ensemble Fitting', icon: Cpu },
              { step: '05', title: 'Visualization', desc: 'Plotly Curve Rendering', icon: LineChart },
              { step: '06', title: 'Export Results', desc: 'Metrics & Data CSV', icon: Download }
            ].map((item, idx) => {
              const IconComp = item.icon;
              return (
                <div key={idx} style={{ padding: '1rem 0.5rem' }}>
                  <div style={{
                    width: '44px',
                    height: '44px',
                    borderRadius: '50%',
                    backgroundColor: 'var(--color-bg)',
                    border: '1px solid var(--color-border)',
                    display: 'flex',
                    alignItems: 'center',
                    justify: 'center',
                    margin: '0 auto 1rem auto',
                    color: 'var(--color-primary)',
                    fontWeight: 700,
                    fontSize: '0.85rem'
                  }}>
                    <IconComp size={20} color="var(--color-accent)" />
                  </div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-accent)', marginBottom: '0.2rem' }}>
                    STEP {item.step}
                  </div>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--color-primary)', marginBottom: '0.25rem' }}>
                    {item.title}
                  </h4>
                  <p style={{ fontSize: '0.78rem', color: 'var(--color-secondary)' }}>
                    {item.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <Footer />
    </div>
  );
};
