import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? (import.meta.env.VITE_API_BASE_URL.endsWith('/api') ? import.meta.env.VITE_API_BASE_URL : `${import.meta.env.VITE_API_BASE_URL}/api`)
  : '/api';

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
  paramVal?: number
): Promise<PointPredictResponse> => {
  const response = await axios.post<PointPredictResponse>(`${API_BASE}/predict/point`, {
    file_id: fileId,
    sheet_name: sheetName,
    curve_type: curveType,
    x_val: xVal,
    param_val: paramVal,
  });
  return response.data;
};

export const predictCurveOverlay = async (
  fileId: string,
  sheetName: string,
  curveType: string,
  paramVal: number
): Promise<CurvePredictResponse> => {
  const response = await axios.post<CurvePredictResponse>(`${API_BASE}/predict/curve`, {
    file_id: fileId,
    sheet_name: sheetName,
    curve_type: curveType,
    param_val: paramVal,
  });
  return response.data;
};
