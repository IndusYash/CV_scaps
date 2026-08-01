import sys
import unittest
import math
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from app.readers.excel_reader import ExcelInputReader
from app.pipeline import UnifiedParserPipeline
from app.domain.ir import IntermediateRepresentationV2

class TestPhase1IRInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wb_path = backend_dir.parent / "Ca3PI3.xlsx"
        if not wb_path.exists():
            wb_path = backend_dir.parent / "Ca3PI3 copy.xlsx"
        
        with open(wb_path, "rb") as f:
            contents = f.read()
            
        cls.pipeline = UnifiedParserPipeline()
        cls.audit_result, cls.parsed_data = cls.pipeline.parse_workbook(
            wb_path.name, contents, "test_phase1_session"
        )
        cls.doc_model = ExcelInputReader().read(wb_path.name, contents)

    def test_invariant_1_curve_data_purity_qe_tdd(self):
        """INVARIANT 1: No descriptor cells (1e6, 1e8) may enter QE target matrix."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        self.assertIsNotNone(sheet_audit, "QE (T+DD) sheet must be present.")
        
        for block in sheet_audit.blocks:
            ir = block.intermediate_representation
            if not ir or not ir.targets:
                continue
            for target in ir.targets:
                if target.name == "QE" and target.dense_matrix:
                    for row in target.dense_matrix:
                        for val in row:
                            if val is not None and not math.isnan(val):
                                self.assertLess(
                                    val, 1000.0,
                                    f"Descriptor value {val} leaked into QE target vector in block {block.block_id}"
                                )

    def test_invariant_1_curve_data_purity_qe_idd(self):
        """INVARIANT 1: No descriptor cells may enter QE (IDD) target matrix."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (IDD)"), None)
        self.assertIsNotNone(sheet_audit, "QE (IDD) sheet must be present.")
        
        for block in sheet_audit.blocks:
            ir = block.intermediate_representation
            if not ir or not ir.targets:
                continue
            for target in ir.targets:
                if target.name == "QE" and target.dense_matrix:
                    for row in target.dense_matrix:
                        for val in row:
                            if val is not None and not math.isnan(val):
                                self.assertLess(
                                    val, 1000.0,
                                    f"Descriptor value {val} leaked into QE target vector in block {block.block_id}"
                                )

    def test_invariant_2_trace_identity_illumination(self):
        """INVARIANT 2: Illumination series (AM1.5G, 400nm, 500nm, 600nm) retain distinct trace identities and shape [63, 4]."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "Illumination"), None)
        self.assertIsNotNone(sheet_audit, "Illumination sheet must be present.")
        
        block = sheet_audit.blocks[0]
        ir = block.intermediate_representation
        self.assertIsNotNone(ir)
        self.assertTrue(len(ir.targets) > 0)
        target = ir.targets[0]
        
        # Verify shape is strictly [63, 4]
        self.assertEqual(target.data_shape, [63, 4], f"Target shape must be [63, 4], got {target.data_shape}")
        
        # Check column_labels on target
        labels = getattr(target, "column_labels", [])
        self.assertEqual(len(labels), 4, f"Target must preserve exactly 4 column labels, got {labels}")
        self.assertEqual(labels, ['AM 1.5G', '400 nm', '500 nm', '600 nm'])

    def test_invariant_3_physical_parameter_preservation_qe_tdd(self):
        """INVARIANT 3: Thickness and Defect Density banners survive as physical parameter metadata across blocks."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "QE (T+DD)"), None)
        self.assertIsNotNone(sheet_audit)
        
        for b in sheet_audit.blocks[:5]:
            ir = b.intermediate_representation
            self.assertIsNotNone(ir)
            param_names = [p.name for p in ir.parameters]
            self.assertTrue(
                any("Thickness" in p or "Defect" in p for p in param_names),
                f"Physical parameters (Thickness/Defect Density) must survive in {b.block_id}, got: {param_names}"
            )

    def test_invariant_4_dependent_variable_separation_optimized(self):
        """INVARIANT 4: QE, Responsivity, Detectivity remain separate dependent variables in IR."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "Optimized"), None)
        self.assertIsNotNone(sheet_audit)
        
        target_block = None
        for b in sheet_audit.blocks:
            ir = b.intermediate_representation
            if ir and any(t.name in ["QE", "Responsivity", "Detectivity"] for t in ir.targets):
                target_block = b
                break
                
        self.assertIsNotNone(target_block, "Multi-target block in Optimized sheet must exist.")
        ir = target_block.intermediate_representation
        target_names = [t.name for t in ir.targets]
        
        self.assertTrue(
            len(ir.targets) >= 2,
            f"Distinct physical quantities must be separate DependentVariables. Got targets: {target_names}"
        )

    def test_existing_unoptimized_jv_regression(self):
        """Regression Check: Unoptimized J-V parsing remains intact."""
        sheet_audit = next((s for s in self.audit_result.sheets if s.sheet_name == "Unoptimized"), None)
        self.assertIsNotNone(sheet_audit)
        b0 = sheet_audit.blocks[0]
        ir = b0.intermediate_representation
        self.assertIsNotNone(ir)
        self.assertEqual(ir.axes[0].name, "Voltage")
        self.assertEqual(ir.targets[0].name, "Current")
        self.assertTrue(len(ir.axes[0].values) > 10)

if __name__ == "__main__":
    unittest.main()
