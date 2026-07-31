import React, { useState, useEffect } from 'react';
import { Layers, RefreshCw, Cpu, Award } from 'lucide-react';
import { FileUploader } from './components/FileUploader';
import { TrainingControls } from './components/TrainingControls';
import { MetricsPanel } from './components/MetricsPanel';
import { CurveVisualizer } from './components/CurveVisualizer';
import { PointPredictor } from './components/PointPredictor';
import { ComparisonOverlay } from './components/ComparisonOverlay';
import { PhysicsPanel } from './components/PhysicsPanel';
import { WorkbookSummary } from './components/WorkbookSummary';
import { Footer } from './components/Footer';
import { trainModels, fetchCurvesData, UploadResponse, TrainResponse } from './api/client';

const SESSION_KEY = 'SOLAR_SIMULATION_SESSION_V1';

export const App: React.FC = () => {
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [activeSheet, setActiveSheet] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'sheet' | 'summary'>('sheet');

  // Hyperparameters
  const [testSize, setTestSize] = useState<number>(0.2);
  const [nEstimators, setNEstimators] = useState<number>(200);
  const [randomState, setRandomState] = useState<number>(42);

  // State per trained sheet
  const [trainingResults, setTrainingResults] = useState<Record<string, TrainResponse>>({});
  const [curvesData, setCurvesData] = useState<Record<string, Record<string, any[]>>>({});
  const [loadingTrain, setLoadingTrain] = useState<boolean>(false);
  const [loadingTrainAll, setLoadingTrainAll] = useState<boolean>(false);

  // SESSION RESTORATION ON REFRESH
  useEffect(() => {
    try {
      const cached = localStorage.getItem(SESSION_KEY);
      if (cached) {
        const parsed = JSON.parse(cached);
        if (parsed.uploadData) {
          setUploadData(parsed.uploadData);
          setActiveSheet(parsed.activeSheet || parsed.uploadData.sheets?.[0]?.sheet_name || null);
          if (parsed.trainingResults) setTrainingResults(parsed.trainingResults);
          if (parsed.curvesData) setCurvesData(parsed.curvesData);
          if (parsed.viewMode) setViewMode(parsed.viewMode);
        }
      }
    } catch (e) {
      console.error('Failed to restore session:', e);
    }
  }, []);

  // SAVE SESSION ON STATE CHANGES
  useEffect(() => {
    if (uploadData) {
      try {
        localStorage.setItem(SESSION_KEY, JSON.stringify({
          uploadData,
          activeSheet,
          trainingResults,
          curvesData,
          viewMode
        }));
      } catch (e) {
        console.error('Failed to cache session:', e);
      }
    }
  }, [uploadData, activeSheet, trainingResults, curvesData, viewMode]);

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadData(data);
    setTrainingResults({});
    setCurvesData({});
    setViewMode('sheet');
    if (data.sheets && data.sheets.length > 0) {
      setActiveSheet(data.sheets[0].sheet_name);
    }
  };

  const handleClearSession = () => {
    localStorage.removeItem(SESSION_KEY);
    setUploadData(null);
    setActiveSheet(null);
    setTrainingResults({});
    setCurvesData({});
    setViewMode('sheet');
  };

  const handleTrain = async () => {
    if (!uploadData || !activeSheet) return;
    setLoadingTrain(true);
    try {
      const result = await trainModels(uploadData.file_id, activeSheet, testSize, nEstimators, randomState);
      setTrainingResults(prev => ({ ...prev, [activeSheet]: result }));

      const curves = await fetchCurvesData(uploadData.file_id, activeSheet);
      setCurvesData(prev => ({ ...prev, [activeSheet]: curves.curves }));
    } catch (err: any) {
      console.error(err);
      alert('Training failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoadingTrain(false);
    }
  };

  // 1-CLICK BATCH TRAINING ACROSS ALL WORKSHEETS
  const handleTrainAll = async () => {
    if (!uploadData || !uploadData.sheets || uploadData.sheets.length === 0) return;
    setLoadingTrainAll(true);
    try {
      for (const s of uploadData.sheets) {
        const sName = s.sheet_name;
        const result = await trainModels(uploadData.file_id, sName, testSize, nEstimators, randomState);
        const curves = await fetchCurvesData(uploadData.file_id, sName);

        setTrainingResults(prev => ({ ...prev, [sName]: result }));
        setCurvesData(prev => ({ ...prev, [sName]: curves.curves }));
      }
    } catch (err: any) {
      console.error(err);
      alert('Batch training encountered an error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoadingTrainAll(false);
    }
  };

  const currentSheetSummary = uploadData?.sheets?.find(s => s.sheet_name === activeSheet);
  const currentResult = activeSheet ? trainingResults[activeSheet] : undefined;
  const currentCurves = activeSheet ? curvesData[activeSheet] : undefined;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--color-bg)' }}>
      {/* ELEGANT STICKY NAVBAR WITH OFFICIAL MMMUT LOGO */}
      <header style={{
        height: '72px',
        backgroundColor: '#FFFFFF',
        borderBottom: '1px solid var(--color-border)',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div className="max-content" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          {/* Left Side: Official MMMUT Crest Logo & Department */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.9rem' }}>
            <img
              src="/mmmut_logo.png"
              alt="MMMUT Gorakhpur Logo"
              style={{ height: '44px', width: 'auto', objectFit: 'contain' }}
            />
            <div>
              <div style={{ fontSize: '0.98rem', fontWeight: 700, color: 'var(--color-primary)', lineHeight: 1.2 }}>
                Computer Vision Laboratory
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--color-secondary)', lineHeight: 1.2 }}>
                Department of Computer Science & Engineering • MMMUT Gorakhpur
              </div>
            </div>
          </div>

          {/* Right Side: Supervisor & Version */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.84rem', color: 'var(--color-secondary)' }}>
              <Award size={16} color="var(--color-accent)" />
              <span>Supervisor: <strong style={{ color: 'var(--color-primary)' }}>Dr. Satvik Vats</strong></span>
            </div>

            <span style={{ color: '#CBD5E1' }}>|</span>

            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-primary)', backgroundColor: 'var(--color-bg)', padding: '0.25rem 0.6rem', borderRadius: '6px', border: '1px solid var(--color-border)' }}>
              Version 1.0
            </span>

            {uploadData && (
              <button
                className="btn-secondary"
                onClick={handleClearSession}
                style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
              >
                <RefreshCw size={14} />
                <span>Upload New Dataset</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* MAIN CONTENT AREA */}
      {!uploadData ? (
        /* Full-Page Landing View */
        <FileUploader onUploadSuccess={handleUploadSuccess} />
      ) : (
        /* Workspace Dashboard View */
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
          <div style={{ display: 'flex', flex: 1 }}>
            {/* Sidebar */}
            <aside style={{
              width: '295px',
              backgroundColor: '#FFFFFF',
              borderRight: '1px solid var(--color-border)',
              padding: '1.5rem 1.25rem',
              display: 'flex',
              flexDirection: 'column'
            }}>
              {/* Batch Action: Train All Sheets */}
              <button
                className="btn-primary"
                onClick={handleTrainAll}
                disabled={loadingTrainAll}
                style={{
                  width: '100%',
                  marginBottom: '1.25rem',
                  padding: '0.75rem 1rem',
                  fontSize: '0.88rem',
                  justifyContent: 'center'
                }}
              >
                <span>{loadingTrainAll ? 'Training All Sheets...' : 'Train All Worksheets'}</span>
              </button>

              {/* View Switcher Tabs */}
              <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '1.25rem', backgroundColor: 'var(--color-bg)', padding: '0.25rem', borderRadius: '8px', border: '1px solid var(--color-border)' }}>
                <button
                  onClick={() => setViewMode('sheet')}
                  style={{
                    flex: 1,
                    padding: '0.45rem',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: viewMode === 'sheet' ? '#FFFFFF' : 'transparent',
                    color: viewMode === 'sheet' ? 'var(--color-primary)' : 'var(--color-secondary)',
                    cursor: 'pointer',
                    boxShadow: viewMode === 'sheet' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                  }}
                >
                  Worksheets ({uploadData.sheets.length})
                </button>
                <button
                  onClick={() => setViewMode('summary')}
                  style={{
                    flex: 1,
                    padding: '0.45rem',
                    fontSize: '0.8rem',
                    fontWeight: 700,
                    borderRadius: '6px',
                    border: 'none',
                    backgroundColor: viewMode === 'summary' ? '#FFFFFF' : 'transparent',
                    color: viewMode === 'summary' ? 'var(--color-accent)' : 'var(--color-secondary)',
                    cursor: 'pointer',
                    boxShadow: viewMode === 'summary' ? '0 1px 2px rgba(0,0,0,0.05)' : 'none'
                  }}
                >
                  Summary Table
                </button>
              </div>

              {/* Worksheet List */}
              <div style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--color-secondary)', marginBottom: '0.8rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Select Parameter Sweep
              </div>

              {uploadData.sheets.map(s => {
                const res = trainingResults[s.sheet_name];
                const isTrained = !!res;
                const isCurrent = viewMode === 'sheet' && activeSheet === s.sheet_name;

                // Compute R2 score badge display
                const jvR2 = res?.metrics?.jv?.r2;
                const qeR2 = res?.metrics?.qe?.r2;
                const avgR2 = jvR2 && qeR2 ? ((jvR2 + qeR2) / 2).toFixed(3) : jvR2 ? jvR2.toFixed(3) : null;

                return (
                  <button
                    key={s.sheet_name}
                    onClick={() => {
                      setActiveSheet(s.sheet_name);
                      setViewMode('sheet');
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '0.75rem 0.9rem',
                      marginBottom: '0.5rem',
                      borderRadius: '8px',
                      border: `1px solid ${isCurrent ? 'var(--color-primary)' : 'var(--color-border)'}`,
                      backgroundColor: isCurrent ? 'var(--color-bg)' : '#FFFFFF',
                      color: isCurrent ? 'var(--color-primary)' : 'var(--color-secondary)',
                      fontWeight: isCurrent ? 700 : 400,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justify: 'space-between',
                      transition: 'all 200ms ease'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Layers size={16} color={isCurrent ? 'var(--color-primary)' : 'var(--color-secondary)'} />
                      <span style={{ fontSize: '0.88rem' }}>{s.sheet_name}</span>
                    </div>

                    {isTrained ? (
                      <span style={{ fontSize: '0.7rem', backgroundColor: '#F0FDF4', color: 'var(--color-success)', border: '1px solid #BBF7D0', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: 700 }}>
                        {avgR2 ? `R²: ${avgR2}` : 'Trained'}
                      </span>
                    ) : (
                      <span style={{ fontSize: '0.7rem', color: 'var(--color-secondary)', backgroundColor: 'var(--color-bg)', padding: '0.15rem 0.45rem', borderRadius: '4px' }}>
                        Ready
                      </span>
                    )}
                  </button>
                );
              })}
            </aside>

            {/* Dashboard Canvas */}
            <main style={{ flex: 1, padding: '2rem 2.5rem' }}>
              {viewMode === 'summary' ? (
                /* Multi-Sheet Comparison Dashboard View */
                <WorkbookSummary
                  uploadData={uploadData}
                  trainingResults={trainingResults}
                  onSelectSheet={(sName) => {
                    setActiveSheet(sName);
                    setViewMode('sheet');
                  }}
                  onTrainAll={handleTrainAll}
                  loadingTrainAll={loadingTrainAll}
                />
              ) : (
                /* Individual Sheet Workspace View */
                <div>
                  {/* Sheet Header Card */}
                  <div className="ui-card" style={{ padding: '1.5rem', marginBottom: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <h2 className="text-card-title" style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>
                        {activeSheet} Physical Analysis & Surrogate Modeling
                      </h2>
                      <p className="text-muted" style={{ margin: 0 }}>
                        Swept Parameter: <strong style={{ color: 'var(--color-primary)' }}>{currentSheetSummary?.detected_parameter_name}</strong> ({currentSheetSummary?.detected_parameter_unit || 'Unitless'})
                      </p>
                    </div>
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                      {currentSheetSummary?.curve_types_found.map(ct => (
                        <span key={ct} style={{ backgroundColor: 'var(--color-bg)', border: '1px solid var(--color-border)', color: 'var(--color-primary)', padding: '0.35rem 0.75rem', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 700, fontFamily: 'monospace' }}>
                          {ct.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Controls */}
                  <TrainingControls
                    testSize={testSize}
                    setTestSize={setTestSize}
                    nEstimators={nEstimators}
                    setNEstimators={setNEstimators}
                    randomState={randomState}
                    setRandomState={setRandomState}
                    onTrain={handleTrain}
                    loading={loadingTrain}
                  />

                  {/* Content */}
                  {!currentResult || !activeSheet ? (
                    <div className="ui-card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
                      <Cpu color="var(--color-primary)" size={48} style={{ marginBottom: '1.25rem' }} />
                      <h3 className="text-card-title" style={{ marginBottom: '0.5rem' }}>
                        Surrogate Model Ready for Fitting
                      </h3>
                      <p className="text-muted" style={{ maxWidth: '520px', margin: '0 auto 2rem auto' }}>
                        Adjust the Random Forest hyperparameters above and click <strong>"Train Physics Models"</strong> to fit regressors across JV, QE, and Responsivity curves.
                      </p>
                      <button className="btn-primary" onClick={handleTrain} disabled={loadingTrain}>
                        {loadingTrain ? 'Fitting Regressors...' : 'Train Physics Models'}
                      </button>
                    </div>
                  ) : (
                    <div>
                      {/* Accuracy Metrics */}
                      <MetricsPanel metrics={currentResult.metrics} />

                      {/* Raw Spectral Curves */}
                      <CurveVisualizer
                        curvesData={currentCurves || null}
                        parameterName={currentResult.parameter_name}
                      />

                      {/* Live Point Predictor */}
                      <PointPredictor
                        fileId={uploadData.file_id}
                        sheetName={activeSheet}
                        availableCurves={currentResult.available_curve_types}
                        parameterName={currentResult.parameter_name}
                        parameterValues={currentResult.parameter_values}
                        xRanges={currentResult.x_ranges}
                      />

                      {/* Curve Comparison Overlay */}
                      <ComparisonOverlay
                        fileId={uploadData.file_id}
                        sheetName={activeSheet}
                        availableCurves={currentResult.available_curve_types}
                        parameterName={currentResult.parameter_name}
                        parameterValues={currentResult.parameter_values}
                      />

                      {/* Physical Theory Accordion */}
                      <PhysicsPanel
                        sheetName={activeSheet}
                        parameterName={currentResult.parameter_name}
                      />
                    </div>
                  )}
                </div>
              )}
            </main>
          </div>

          {/* ACADEMIC FOOTER ALSO PRESENT ON DASHBOARD SCREEN */}
          <Footer />
        </div>
      )}
    </div>
  );
};
export default App;
