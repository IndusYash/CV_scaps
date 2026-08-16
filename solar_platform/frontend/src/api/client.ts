import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? (import.meta.env.VITE_API_BASE_URL.endsWith('/api') ? import.meta.env.VITE_API_BASE_URL : `${import.meta.env.VITE_API_BASE_URL}/api`)
  : '/api';

export interface ParameterDefinition {
  name: string;
  unit: string;
  values: number[];
}

export interface TracePayload {
  trace_id: string;
  label: string;
  parameters: Record<string, number>;
  x: number[];
  y: number[];
  source_block_id: string;
}

export interface ExperimentPayload {
  experiment_id: string;
  sheet_name: string;
  semantic_type: string;
  human_label: string;
  axis: { name: string; unit: string };
  target: { name: string; unit: string };
  parameters: ParameterDefinition[];
  traces: TracePayload[];
  source_block_ids: string[];
}

export interface ExperimentSummary {
  experiment_id: string;
  sheet_name: string;
  semantic_type: string;
  human_label: string;
  axis_name: string;
  axis_unit: string;
  target_name: string;
  target_unit: string;
  parameter_names: string[];
  trace_count: number;
  source_block_count: number;
}

export interface BlockSummary {
  block_id: string;
  category: string;
  semantic_label: string;
  human_label: string;
  axis_name: string;
  axis_unit: string;
  target_name: string;
  target_unit: string;
  parameters: string[];
  row_count: number;
  confidence: number;
  scientific_validity: number;
  warnings: string[];
  summary_metrics?: Record<string, Record<string, number>>;
}

export interface SheetSummary {
  sheet_name: string;
  row_count: number;
  col_count: number;
  columns: string[];
  detected_parameter_name: string;
  detected_parameter_unit: string;
  detected_x_axis: string;
  curve_types_found: string[];
  is_single_curve_mode: boolean;
  blocks: BlockSummary[];
  experiments: ExperimentSummary[];
}

export interface UploadResponse {
  filename: string;
  file_id: string;
  sheets: SheetSummary[];
}

export interface MetricPair {
  r2: number;
  mae: number;
}

export interface TrainResponse {
  sheet_name: string;
  metrics: Record<string, MetricPair>;
  available_curve_types: string[];
  parameter_name: string;
  parameter_values: number[];
  x_ranges: Record<string, { min: number; max: number }>;
}

export interface CurvesDataResponse {
  sheet_name: string;
  curves: Record<string, any[]>;
  experiments: ExperimentPayload[];
}

export interface PointPredictResponse {
  curve_type: string;
  predicted_value: number;
}

export interface CurvePredictResponse {
  curve_type: string;
  param_val: number;
  actual_x: number[];
  actual_y: number[];
  pred_x: number[];
  pred_y: number[];
}

export const uploadExcelFile = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await axios.post<UploadResponse>(`${API_BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const trainModels = async (
  fileId: string,
  sheetName: string,
  testSize: number = 0.2,
  nEstimators: number = 200,
  randomState: number = 42
): Promise<TrainResponse> => {
  const response = await axios.post<TrainResponse>(`${API_BASE}/train`, {
    file_id: fileId,
    sheet_name: sheetName,
    test_size: testSize,
    n_estimators: nEstimators,
    random_state: randomState,
  });
  return response.data;
};

export const fetchCurvesData = async (fileId: string, sheetName: string): Promise<CurvesDataResponse> => {
  const response = await axios.post<CurvesDataResponse>(`${API_BASE}/curves`, {
    file_id: fileId,
    sheet_name: sheetName,
  });
  return response.data;
};

export const predictPoint = async (
  fileId: string,
  sheetName: string,
  curveType: string,
  xVal: number,
  paramVal?: number,
  paramVals?: Record<string, number>
): Promise<PointPredictResponse> => {
  const response = await axios.post<PointPredictResponse>(`${API_BASE}/predict/point`, {
    file_id: fileId,
    sheet_name: sheetName,
    experiment_id: curveType,
    target_name: curveType,
    curve_type: curveType,
    x_val: xVal,
    param_val: paramVal,
    param_vals: paramVals,
  });
  return response.data;
};

export const predictCurveOverlay = async (
  fileId: string,
  sheetName: string,
  curveType: string,
  paramVal: number,
  paramVals?: Record<string, number>
): Promise<CurvePredictResponse> => {
  const response = await axios.post<CurvePredictResponse>(`${API_BASE}/predict/curve`, {
    file_id: fileId,
    sheet_name: sheetName,
    experiment_id: curveType,
    target_name: curveType,
    curve_type: curveType,
    param_val: paramVal,
    param_vals: paramVals,
  });
  return response.data;
};

// --- V1 persistent DB & Global ML client API ---

export const fetchDatabaseStatus = async (): Promise<{ datasets_count: number; derived_metrics_count: number }> => {
  const response = await axios.get(`${API_BASE}/database/status`);
  return response.data;
};

export const fetchDatasets = async (scientificType?: string): Promise<any[]> => {
  const url = scientificType ? `${API_BASE}/datasets?scientific_type=${scientificType}` : `${API_BASE}/datasets`;
  const response = await axios.get(url);
  return response.data;
};

export const trainGlobalRandomForest = async (
  material?: string,
  testSize: number = 0.25,
  randomState: number = 42,
  validationMode: string = "interpolation",
  testWorkbook?: string
): Promise<any> => {
  const params = new URLSearchParams();
  if (material) params.append("material", material);
  params.append("test_size", testSize.toString());
  params.append("random_state", randomState.toString());
  params.append("validation_mode", validationMode);
  if (testWorkbook) params.append("test_workbook", testWorkbook);

  const response = await axios.post(`${API_BASE}/ml/train/random-forest?${params.toString()}`);
  return response.data;
};

export const fetchGlobalMLResults = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE}/ml/results`);
  return response.data;
};

