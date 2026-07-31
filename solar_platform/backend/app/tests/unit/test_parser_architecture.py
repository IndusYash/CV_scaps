from __future__ import annotations

import os
import unittest
from app.pipeline import UnifiedParserPipeline
from app.readers.excel_reader import ExcelInputReader
from app.semantics.registry import SemanticRegistry


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

    def test_semantic_registry_evaluation(self):
        registry = SemanticRegistry()
        self.assertTrue(len(registry._plugins) >= 3)

    def test_unified_pipeline_end_to_end(self):
        pipeline = UnifiedParserPipeline()
        sample_path = os.path.join("..", "sample_data.xlsx")
        if not os.path.exists(sample_path):
            sample_path = "../sample_data.xlsx"

        with open(sample_path, "rb") as f:
            contents = f.read()

        audit_result, downstream_data = pipeline.parse_workbook("sample_data.xlsx", contents, "test_file_id_001")
        self.assertEqual(audit_result.filename, "sample_data.xlsx")
        self.assertTrue(audit_result.total_blocks_found > 0)
        self.assertTrue(len(downstream_data) > 0)
        self.assertIn("ETL", downstream_data)
        self.assertIn("jv", downstream_data["ETL"])


if __name__ == "__main__":
    unittest.main()
