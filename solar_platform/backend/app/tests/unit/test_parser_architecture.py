from __future__ import annotations

import os
import unittest
import numpy as np
import pandas as pd
from app.adapters.rf_adapter import RandomForestMLAdapter
from app.domain.audit import BoundingBox, DataBlockAudit
from app.domain.document import DocumentCell, DocumentSheet, CellDataType
from app.domain.ir_generator import IRGeneratorEngine
from app.exceptions import UnsupportedFormatException
from app.pipeline import UnifiedParserPipeline
from app.readers.excel_reader import ExcelInputReader
from app.readers.factory import ReaderFactory
from app.semantics.registry import SemanticRegistry
from app.services.ml_engine import (
    _SESSION_STORE,
    get_trained_session,
    predict_point,
    train_sheet_models,
)


class TestParserArchitecture(unittest.TestCase):
    def test_excel_reader_loading(self):
        reader = ExcelInputReader()
        sample_path = os.path.join("..", "sample_data.xlsx")
        if not os.path.exists(sample_path):
            sample_path = "../sample_data.xlsx"

        with open(sample_path, "rb") as f:
            contents = f.read()

        doc_model = reader.read("sample_data.xlsx", contents)
        self.assertEqual(doc_model.document_name, "sample_data.xlsx")
        self.assertTrue(len(doc_model.sheets) > 0)
        self.assertEqual(doc_model.sheets[0].sheet_name, "ETL")

    def test_reader_factory_resolution(self):
        csv_bytes = b"Voltage,Current\n0.0,0.01\n0.1,0.02\n"
        reader = ReaderFactory.get_reader("data.csv", csv_bytes)
        doc = reader.read("data.csv", csv_bytes)
        self.assertEqual(doc.sheets[0].sheet_name, "CSV_Data")

    def test_unsupported_format_exception(self):
        with self.assertRaises(UnsupportedFormatException):
            ReaderFactory.get_reader("invalid.txt", b"some text content")

    def test_semantic_registry_evaluation(self):
        registry = SemanticRegistry()
        self.assertTrue(len(registry._plugins) >= 6)

    def test_absorption_vs_qe_discrimination(self):
        """Verifies that Absorption Coefficient datasets are tagged ABSORPTION, not QE."""
        row_0 = [
            DocumentCell(row_index=0, col_index=0, raw_value="Lambda(nm)", formatted_value="Lambda(nm)", data_type=CellDataType.STRING),
            DocumentCell(row_index=0, col_index=1, raw_value="alpha(1/cm)", formatted_value="alpha(1/cm)", data_type=CellDataType.STRING),
        ]
        row_1 = [
            DocumentCell(row_index=1, col_index=0, raw_value=300.0, formatted_value="300.0", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=1, col_index=1, raw_value=1.5e5, formatted_value="1.5e5", data_type=CellDataType.NUMBER),
        ]

        sheet = DocumentSheet(
            sheet_name="absorption",
            sheet_index=0,
            row_count=2,
            col_count=2,
            cells=[row_0, row_1],
        )

        block = DataBlockAudit(
            block_id="b1",
            bounding_box=BoundingBox(start_row=0, end_row=1, start_col=0, end_col=1),
        )

        ir_engine = IRGeneratorEngine()
        ir = ir_engine.generate_ir_for_block(sheet, block)
        registry = SemanticRegistry()
        label = registry.evaluate_and_tag(ir, block)

        self.assertEqual(ir.targets[0].name, "Absorption")
        self.assertEqual(label, "ABSORPTION")

    def test_multi_parameter_sweep_ir_generation(self):
        """Verifies multi-parameter extraction (Thickness=100nm, Nt=1e14) into IR parameters."""
        row_0 = [
            DocumentCell(row_index=0, col_index=0, raw_value="V(V)", formatted_value="V(V)", data_type=CellDataType.STRING),
            DocumentCell(row_index=0, col_index=1, raw_value="t=100nm_Nt=1e14", formatted_value="t=100nm_Nt=1e14", data_type=CellDataType.STRING),
            DocumentCell(row_index=0, col_index=2, raw_value="t=200nm_Nt=1e15", formatted_value="t=200nm_Nt=1e15", data_type=CellDataType.STRING),
        ]
        row_1 = [
            DocumentCell(row_index=1, col_index=0, raw_value=0.0, formatted_value="0.0", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=1, col_index=1, raw_value=12.5, formatted_value="12.5", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=1, col_index=2, raw_value=14.1, formatted_value="14.1", data_type=CellDataType.NUMBER),
        ]
        row_2 = [
            DocumentCell(row_index=2, col_index=0, raw_value=0.5, formatted_value="0.5", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=2, col_index=1, raw_value=10.2, formatted_value="10.2", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=2, col_index=2, raw_value=11.8, formatted_value="11.8", data_type=CellDataType.NUMBER),
        ]

        sheet = DocumentSheet(
            sheet_name="Thick+DD(PVK)",
            sheet_index=0,
            row_count=3,
            col_count=3,
            cells=[row_0, row_1, row_2],
        )

        block = DataBlockAudit(
            block_id="b1",
            bounding_box=BoundingBox(start_row=0, end_row=2, start_col=0, end_col=2),
        )

        ir_engine = IRGeneratorEngine()
        ir = ir_engine.generate_ir_for_block(sheet, block)

        self.assertEqual(len(ir.axes), 1)
        self.assertEqual(ir.axes[0].name, "Voltage")
        self.assertEqual(len(ir.parameters), 2)
        self.assertEqual(ir.parameters[0].name, "Thickness")
        self.assertEqual(ir.parameters[1].name, "Defect_Density")

        adapter = RandomForestMLAdapter()
        df_long = adapter.transform_to_long_dataframe(ir)
        self.assertIn("Voltage", df_long.columns)
        self.assertIn("Thickness", df_long.columns)
        self.assertIn("Defect_Density", df_long.columns)
        self.assertIn("Current", df_long.columns)

    def test_single_device_optimized_run(self):
        """Verifies parsing single device run ('Optimized') without length mismatch crashes."""
        row_0 = [
            DocumentCell(row_index=0, col_index=0, raw_value="v(V)", formatted_value="v(V)", data_type=CellDataType.STRING),
            DocumentCell(row_index=0, col_index=1, raw_value="j(mA/cm^2)", formatted_value="j(mA/cm^2)", data_type=CellDataType.STRING),
        ]
        row_1 = [
            DocumentCell(row_index=1, col_index=0, raw_value=0.0, formatted_value="0.0", data_type=CellDataType.NUMBER),
            DocumentCell(row_index=1, col_index=1, raw_value=22.1, formatted_value="22.1", data_type=CellDataType.NUMBER),
        ]

        sheet = DocumentSheet(
            sheet_name="Optimized",
            sheet_index=0,
            row_count=2,
            col_count=2,
            cells=[row_0, row_1],
        )

        block = DataBlockAudit(
            block_id="b1",
            bounding_box=BoundingBox(start_row=0, end_row=1, start_col=0, end_col=1),
        )

        ir_engine = IRGeneratorEngine()
        ir = ir_engine.generate_ir_for_block(sheet, block)
        self.assertEqual(ir.axes[0].name, "Voltage")
        self.assertEqual(ir.targets[0].name, "Current")

    def test_multi_block_model_session_isolation(self):
        """REGRESSION TEST: Verifies 3 independent JV blocks + 1 QE block on one worksheet preserve independent models without overwriting."""
        file_id = "test_multi_block_001"
        sheet_name = "MultiBlockSheet"

        df_b1 = pd.DataFrame({"Voltage": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "Parameter_1": [10, 10, 10, 10, 10, 10], "Current": [20.0, 20.1, 20.2, 20.3, 20.4, 20.5]})
        df_b1.attrs = {"block_id": "block_1", "semantic_label": "jv"}

        df_b2 = pd.DataFrame({"Voltage": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "Temperature": [300, 300, 300, 300, 300, 300], "Current": [35.0, 35.1, 35.2, 35.3, 35.4, 35.5]})
        df_b2.attrs = {"block_id": "block_2", "semantic_label": "jv"}

        df_b3 = pd.DataFrame({"Voltage": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "Thickness": [500, 500, 500, 500, 500, 500], "Current": [50.0, 50.1, 50.2, 50.3, 50.4, 50.5]})
        df_b3.attrs = {"block_id": "block_3", "semantic_label": "jv"}

        df_b4 = pd.DataFrame({"Wavelength": [300, 400, 500, 600, 700, 800], "Thickness": [100, 100, 100, 100, 100, 100], "QE": [80.0, 82.0, 85.0, 88.0, 90.0, 92.0]})
        df_b4.attrs = {"block_id": "block_4", "semantic_label": "qe"}

        parsed_sheet_data = {
            "block_1": df_b1,
            "block_2": df_b2,
            "block_3": df_b3,
            "block_4": df_b4,
        }

        sessions = train_sheet_models(
            file_id=file_id,
            sheet_name=sheet_name,
            df_raw=pd.DataFrame(),
            parsed_sheet_data=parsed_sheet_data,
        )

        # Assert ALL 4 models exist simultaneously
        self.assertEqual(len(sessions), 4)
        self.assertIn("block_1", sessions)
        self.assertIn("block_2", sessions)
        self.assertIn("block_3", sessions)
        self.assertIn("block_4", sessions)

        # Assert independent model retrieval by block_id
        sess1 = get_trained_session(file_id, sheet_name, block_id="block_1")
        sess2 = get_trained_session(file_id, sheet_name, block_id="block_2")
        sess3 = get_trained_session(file_id, sheet_name, block_id="block_3")
        sess4 = get_trained_session(file_id, sheet_name, block_id="block_4")

        self.assertIsNotNone(sess1)
        self.assertIsNotNone(sess2)
        self.assertIsNotNone(sess3)
        self.assertIsNotNone(sess4)

        # Assert distinct predictions without overwriting
        p1 = predict_point(file_id, sheet_name, "jv", block_id="block_1", x_val=0.0, param_val=10.0)
        p2 = predict_point(file_id, sheet_name, "jv", block_id="block_2", x_val=0.0, param_val=300.0)
        p3 = predict_point(file_id, sheet_name, "jv", block_id="block_3", x_val=0.0, param_val=500.0)
        p4 = predict_point(file_id, sheet_name, "qe", block_id="block_4", x_val=300.0, param_val=100.0)

        self.assertAlmostEqual(p1, 20.0, delta=1.0)
        self.assertAlmostEqual(p2, 35.0, delta=1.0)
        self.assertAlmostEqual(p3, 50.0, delta=1.0)
        self.assertAlmostEqual(p4, 86.0, delta=10.0)


if __name__ == "__main__":
    unittest.main()
