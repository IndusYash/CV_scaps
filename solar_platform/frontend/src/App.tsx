import React, { useState } from 'react';
import { UploadResponse, TrainResponse, SheetSummary, BlockSummary, fetchCurvesData, trainModels } from './api/client';
import { FileUploader } from './components/FileUploader';
import { WorkbookSummary } from './components/WorkbookSummary';
import { SummaryMetricsPanel } from './components/SummaryMetricsPanel';
import { TrainingControls } from './components/TrainingControls';
import { MetricsPanel } from './components/MetricsPanel';
import { PointPredictor } from './components/PointPredictor';
import { ComparisonOverlay } from './components/ComparisonOverlay';
import { CurveVisualizer } from './components/CurveVisualizer';
import { Footer } from './components/Footer';
import {
  FileSpreadsheet,
  Layers,
  Activity,
  Award,
  ChevronRight,
  RefreshCw,
  Info,
  ShieldCheck,
  Zap,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import './styles/index.css';

export const App: React.FC = () => {
  const [uploadData, setUploadData] = useState<UploadResponse | null>(null);
  const [selectedSheetName, setSelectedSheetName] = useState<string>('');
  const [selectedBlock, setSelectedBlock] = useState<BlockSummary | null>(null);
  const [trainingResults, setTrainingResults] = useState<Record<string, TrainResponse>>({});
  const [curvesData, setCurvesData] = useState<Record<string, any[]> | null>(null);
  const [loadingTrainAll, setLoadingTrainAll] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'experiment'>('dashboard');

  const handleUploadSuccess = (data: UploadResponse) => {
    setUploadData(data);
    setTrainingResults({});
    setCurvesData(null);
    if (data.sheets.length > 0) {
      const firstSheet = data.sheets[0];
      setSelectedSheetName(firstSheet.sheet_name);
      if (firstSheet.blocks && firstSheet.blocks.length > 0) {
        setSelectedBlock(firstSheet.blocks[0]);
      }
    }
    setActiveTab('dashboard');
  };

  const handleSelectSheetAndBlock = async (sheetName: string, block: BlockSummary) => {
    setSelectedSheetName(sheetName);
    setSelectedBlock(block);
    setActiveTab('experiment');

    if (uploadData?.file_id) {
      try {
        const cData = await fetchCurvesData(uploadData.file_id, sheetName);
        setCurvesData(cData.curves);
      } catch (e) {
        console.error('Failed to fetch curves data', e);
      }
    }
  };

  const handleTrainingComplete = async (res: TrainResponse) => {
    setTrainingResults(prev => ({
      ...prev,
      [selectedSheetName]: res
    }));
    if (uploadData?.file_id && selectedSheetName) {
      try {
        const cData = await fetchCurvesData(uploadData.file_id, selectedSheetName);
        setCurvesData(cData.curves);
      } catch (e) {
        console.error(e);
      }
    }
  };

  const handleTrainAll = async () => {
    if (!uploadData) return;
    setLoadingTrainAll(true);
    try {
      const newResults: Record<string, TrainResponse> = {};
      for (const s of uploadData.sheets) {
        const res = await trainModels(uploadData.file_id, s.sheet_name);
        newResults[s.sheet_name] = res;
      }
      setTrainingResults(newResults);
      if (selectedSheetName) {
        const cData = await fetchCurvesData(uploadData.file_id, selectedSheetName);
        setCurvesData(cData.curves);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingTrainAll(false);
    }
  };

  const currentSheetSummary = uploadData?.sheets.find(s => s.sheet_name === selectedSheetName);
  const currentTraining = trainingResults[selectedSheetName];
  const parameterValues = currentTraining?.parameter_values || [];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#F8FAFC' }}>
      {/* Executive Header Branding Banner */}
      <header style={{ backgroundColor: '#0F172A', color: '#FFFFFF', padding: '1.2rem 2.5rem', borderBottom: '1px solid #1E293B' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ backgroundColor: '#0284C7', padding: '0.6rem', borderRadius: '10px', display: 'flex', alignItems: 'center' }}>
              <Zap size={22} color="#FFFFFF" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 800, margin: 0, letterSpacing: '-0.02em', color: '#FFFFFF' }}>
                SCAPS ML Analytics Platform
              </h1>
              <span style={{ fontSize: '0.78rem', color: '#94A3B8', fontWeight: 500 }}>
                Computer Vision & Materials Physics Research Lab | Department of Physics, MMMUT
              </span>
            </div>
          </div>

          {uploadData && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.2rem' }}>
              <span style={{ fontSize: '0.85rem', color: '#CBD5E1', backgroundColor: '#1E293B', padding: '0.4rem 0.9rem', borderRadius: '6px', border: '1px solid #334155' }}>
                Active File: <strong>{uploadData.filename}</strong>
              </span>
              <button
                onClick={() => setUploadData(null)}
                style={{ backgroundColor: '#334155', color: '#FFFFFF', border: 'none', padding: '0.45rem 0.9rem', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 600, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
              >
                <RefreshCw size={14} /> Upload New File
              </button>
            </div>
          )}
        </div>
      </header>

      {/* Main Workspace Area */}
      <main style={{ flex: 1, maxWidth: '1400px', width: '100%', margin: '0 auto', padding: '2rem 2.5rem' }}>
        {!uploadData ? (
          <div style={{ maxWidth: '850px', margin: '2rem auto' }}>
            <FileUploader onUploadSuccess={handleUploadSuccess} />
          </div>
        ) : (
          <div>
            {/* Top Workspace Navigation Tabs */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2rem', borderBottom: '2px solid #E2E8F0', paddingBottom: '0.5rem' }}>
              <button
                onClick={() => setActiveTab('dashboard')}
                style={{
                  padding: '0.6rem 1.2rem',
                  fontSize: '0.95rem',
                  fontWeight: 700,
                  border: 'none',
                  borderBottom: activeTab === 'dashboard' ? '3px solid #0284C7' : 'none',
                  backgroundColor: 'transparent',
                  color: activeTab === 'dashboard' ? '#0284C7' : '#64748B',
                  cursor: 'pointer'
                }}
              >
                Workbook Hierarchy Overview
              </button>

              {selectedBlock && (
                <button
                  onClick={() => setActiveTab('experiment')}
                  style={{
                    padding: '0.6rem 1.2rem',
                    fontSize: '0.95rem',
                    fontWeight: 700,
                    border: 'none',
                    borderBottom: activeTab === 'experiment' ? '3px solid #0284C7' : 'none',
                    backgroundColor: 'transparent',
                    color: activeTab === 'experiment' ? '#0284C7' : '#64748B',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem'
                  }}
                >
                  <ChevronRight size={16} />
                  Experiment Analysis: {selectedBlock.human_label} ({selectedSheetName})
                </button>
              )}
            </div>

            {/* TAB 1: WORKBOOK DASHBOARD */}
            {activeTab === 'dashboard' && (
              <WorkbookSummary
                uploadData={uploadData}
                trainingResults={trainingResults}
                onSelectSheetAndBlock={handleSelectSheetAndBlock}
                onTrainAll={handleTrainAll}
                loadingTrainAll={loadingTrainAll}
              />
            )}

            {/* TAB 2: DETAILED EXPERIMENT ANALYSIS */}
            {activeTab === 'experiment' && selectedBlock && (
              <div>
                {/* Block Info Metadata Bar */}
                <div className="ui-card" style={{ padding: '1.5rem', marginBottom: '2rem', backgroundColor: '#FFFFFF', border: '1px solid #E2E8F0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#0284C7', textTransform: 'uppercase', marginBottom: '0.2rem' }}>
                        Sheet: '{selectedSheetName}' | Category: {selectedBlock.category}
                      </div>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: 800, margin: 0, color: '#0F172A' }}>
                        {selectedBlock.human_label}
                      </h2>
                    </div>

                    <div style={{ display: 'flex', gap: '1.5rem', fontSize: '0.88rem', color: '#475569' }}>
                      <div><strong>Axis:</strong> {selectedBlock.axis_name} ({selectedBlock.axis_unit})</div>
                      <div><strong>Target:</strong> {selectedBlock.target_name} ({selectedBlock.target_unit})</div>
                      <div><strong>Confidence:</strong> {(selectedBlock.confidence * 100).toFixed(0)}%</div>
                    </div>
                  </div>

                  {selectedBlock.warnings.length > 0 && (
                    <div style={{ marginTop: '1rem', padding: '0.75rem 1rem', backgroundColor: '#FFFBEB', border: '1px solid #FCD34D', borderRadius: '8px', fontSize: '0.85rem', color: '#92400E', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <AlertTriangle size={18} color="#D97706" />
                      <span><strong>Scientific Physics Bounds Notice:</strong> {selectedBlock.warnings.join('; ')}</span>
                    </div>
                  )}
                </div>

                {/* If SUMMARY_METRICS Block: Render Result Cards Table */}
                {selectedBlock.category === 'SUMMARY' || selectedBlock.semantic_label === 'SUMMARY_METRICS' ? (
                  <SummaryMetricsPanel
                    summaryMetrics={selectedBlock.summary_metrics || {}}
                    sheetName={selectedSheetName}
                  />
                ) : (
                  /* Trainable Curve Experiment Pipeline */
                  <div>
                    <TrainingControls
                      fileId={uploadData.file_id}
                      sheetName={selectedSheetName}
                      selectedBlock={selectedBlock}
                      onTrainingComplete={handleTrainingComplete}
                    />

                    {currentTraining && (
                      <MetricsPanel
                        metrics={currentTraining.metrics}
                        selectedBlock={selectedBlock}
                      />
                    )}

                    {currentTraining && (
                      <PointPredictor
                        fileId={uploadData.file_id}
                        sheetName={selectedSheetName}
                        selectedBlock={selectedBlock}
                        parameterValues={parameterValues}
                        xRanges={currentTraining.x_ranges || {}}
                      />
                    )}

                    {currentTraining && (
                      <ComparisonOverlay
                        fileId={uploadData.file_id}
                        sheetName={selectedSheetName}
                        selectedBlock={selectedBlock}
                        parameterValues={parameterValues}
                      />
                    )}

                    <CurveVisualizer
                      curvesData={curvesData}
                      parameterName={currentTraining?.parameter_name || selectedBlock.parameters[0] || 'Parameter'}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default App;
