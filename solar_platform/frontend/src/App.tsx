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
import { trainModels, fetchCurvesData, UploadResponse, TrainResponse, ExperimentPayload } from './api/client';

const SESSION_KEY = 'SOLAR_SIMULATION_SESSION_V2';

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
  const [experimentsData, setExperimentsData] = useState<Record<string, ExperimentPayload[]>>({});
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
          if (parsed.experimentsData) setExperimentsData(parsed.experimentsData);
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
          experimentsData,
          viewMode
        }));
      } catch (e) {
        console.error('Failed to cache session:', e);
      }
    }
  }, [uploadData, activeSheet, trainingResults, curvesData, experimentsData, viewMode]);

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadData(data);
    setTrainingResults({});
    setCurvesData({});
    setExperimentsData({});
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
    setExperimentsData({});
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
      if (curves.experiments) {
        setExperimentsData(prev => ({ ...prev, [activeSheet]: curves.experiments }));
      }
    } catch (err: any) {
      console.error(err);
      if (err.response?.status === 404) {
        alert('Server session expired or local server reloaded. Please re-upload your Excel workbook.');
        handleClearSession();
      } else {
        alert('Training failed: ' + (err.response?.data?.detail || err.message));
      }
    } finally {
      setLoadingTrain(false);
    }
  };

  // Automatically fetch curves when switching sheets if uploadData is ready
  useEffect(() => {
    if (uploadData && activeSheet && !experimentsData[activeSheet]) {
      fetchCurvesData(uploadData.file_id, activeSheet).then(res => {
        if (res.curves) setCurvesData(prev => ({ ...prev, [activeSheet]: res.curves }));
        if (res.experiments) setExperimentsData(prev => ({ ...prev, [activeSheet]: res.experiments }));
      }).catch(err => console.error(err));
    }
  }, [uploadData, activeSheet]);

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
        if (curves.experiments) {
          setExperimentsData(prev => ({ ...prev, [sName]: curves.experiments }));
        }
      }
    } catch (err: any) {
      console.error(err);
      if (err.response?.status === 404) {
        alert('Server session expired or local server reloaded. Please re-upload your Excel workbook.');
        handleClearSession();
      } else {
        alert('Batch training encountered an error: ' + (err.response?.data?.detail || err.message));
      }
    } finally {
      setLoadingTrainAll(false);
    }
  };

  const currentSheetSummary = uploadData?.sheets?.find(s => s.sheet_name === activeSheet);
  const currentResult = activeSheet ? trainingResults[activeSheet] : undefined;
  const currentExperiments = activeSheet ? experimentsData[activeSheet] : undefined;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--color-bg)' }}>
      {/* STICKY NAVBAR */}
      <header style={{
        height: '72px',
        backgroundColor: '#FFFFFF',
        borderBottom: '1px solid var(--color-border)',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div className="max-content" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
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

          <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.84rem', color: 'var(--color-secondary)' }}>
              <Award size={16} color="var(--color-accent)" />
              <span>Supervisor: <strong style={{ color: 'var(--color-primary)' }}>Dr. Satvik Vats</strong></span>
            </div>

            <span style={{ color: '#CBD5E1' }}>|</span>

            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--color-primary)', backgroundColor: 'var(--color-bg)', padding: '0.25rem 0.6rem', borderRadius: '6px', border: '1px solid var(--color-border)' }}>
              Version 2.0 (Phase 3 Integration)
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
        <FileUploader onUploadSuccess={handleUploadSuccess} />
      ) : (
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
                  Summary
                </button>
              </div>

              {viewMode === 'sheet' && (
                <div style={{ flex: 1, overflowY: 'auto' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.6rem' }}>
                    Select Worksheet
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {uploadData.sheets.map((s) => {
                      const isActive = s.sheet_name === activeSheet;
                      const isTrained = !!trainingResults[s.sheet_name];

                      return (
                        <button
                          key={s.sheet_name}
                          onClick={() => setActiveSheet(s.sheet_name)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            width: '100%',
                            padding: '0.65rem 0.85rem',
                            borderRadius: '8px',
                            border: '1px solid',
                            borderColor: isActive ? 'var(--color-accent)' : 'transparent',
                            backgroundColor: isActive ? 'var(--color-accent-light)' : 'transparent',
                            color: isActive ? 'var(--color-accent)' : 'var(--color-primary)',
                            fontWeight: isActive ? 700 : 500,
                            fontSize: '0.86rem',
                            cursor: 'pointer',
                            textAlign: 'left',
                            transition: 'all 150ms ease'
                          }}
                        >
                          <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {s.sheet_name}
                          </span>
                          {isTrained && (
                            <span style={{ fontSize: '0.68rem', backgroundColor: '#DCFCE7', color: 'var(--color-success)', padding: '0.1rem 0.4rem', borderRadius: '4px', fontWeight: 700 }}>
                              Fitted
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </aside>

            {/* Content Area */}
            <main style={{ flex: 1, padding: '2rem 2.5rem', overflowY: 'auto' }}>
              {viewMode === 'summary' ? (
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
                currentSheetSummary && (
                  <div>
                    <div style={{ marginBottom: '2rem' }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--color-accent)', marginBottom: '0.2rem' }}>
                        Worksheet Simulation Dataset
                      </div>
                      <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--color-primary)', margin: 0 }}>
                        Sheet: '{currentSheetSummary.sheet_name}'
                      </h2>
                      <div style={{ fontSize: '0.88rem', color: 'var(--color-secondary)', marginTop: '0.3rem' }}>
                        {currentSheetSummary.row_count} data rows × {currentSheetSummary.col_count} columns | Scientific Experiments: <strong>{currentSheetSummary.experiments?.length || 0}</strong>
                      </div>
                    </div>

                    <PhysicsPanel
                      sheetName={currentSheetSummary.sheet_name}
                      parameterName={currentSheetSummary.detected_parameter_name}
                    />

                    {/* Scientific Curves Visualizer */}
                    <CurveVisualizer experiments={currentExperiments || null} />

                    {/* Hyperparameter Training Controls */}
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

                    {/* Metrics Panel */}
                    {currentResult && (
                      <MetricsPanel metrics={currentResult.metrics} />
                    )}

                    {/* Point Predictor */}
                    {currentResult && (
                      <PointPredictor
                        fileId={uploadData.file_id}
                        sheetName={currentSheetSummary.sheet_name}
                        availableCurves={currentResult.available_curve_types}
                        parameterName={currentResult.parameter_name}
                        parameterValues={currentResult.parameter_values}
                        xRanges={currentResult.x_ranges}
                      />
                    )}

                    {/* Curve Comparison Overlay */}
                    {currentResult && (
                      <ComparisonOverlay
                        fileId={uploadData.file_id}
                        sheetName={currentSheetSummary.sheet_name}
                        availableCurves={currentResult.available_curve_types}
                        parameterName={currentResult.parameter_name}
                        parameterValues={currentResult.parameter_values}
                      />
                    )}
                  </div>
                )
              )}
            </main>
          </div>
        </div>
      )}

      <Footer />
    </div>
  );
};

export default App;
