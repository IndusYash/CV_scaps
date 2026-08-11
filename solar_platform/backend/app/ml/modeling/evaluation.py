"""
app.ml.modeling.evaluation
--------------------------
Evaluation runner for REGIME_A_IN_DOMAIN and REGIME_B_CROSS_MATERIAL split regimes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.ml.curation_pipeline import MLCurationResult
from app.ml.modeling.config import PreprocessorConfig, SplitRegime, TrainingConfig
from app.ml.modeling.dataset import CurvePointDataset
from app.ml.modeling.metrics import CurveEvaluationMetrics
from app.ml.modeling.preprocessing import MLPreprocessor
from app.ml.modeling.trainer import SolarModelTrainer


class EvaluationRunner:
    """Evaluation runner preparing split datasets and benchmarking models on REGIME_A & REGIME_B."""

    def __init__(self, curation_result: MLCurationResult, config: Optional[TrainingConfig] = None):
        self.curation_result = curation_result
        self.config = config or TrainingConfig()

    def prepare_regime_a_datasets(self) -> Dict[str, CurvePointDataset]:
        """Prepare train, val, and test CurvePointDatasets for REGIME_A_IN_DOMAIN."""
        train_fams = set(self.curation_result.regime_a_train_family_ids)
        val_fams = set(self.curation_result.regime_a_val_family_ids)
        test_fams = set(self.curation_result.regime_a_test_family_ids)

        # Lookup family ID for each training curve record
        family_map = {e.dataset_id: e.experiment_family_id for e in self.curation_result.manifest_entries}

        train_recs = []
        val_recs = []
        test_recs = []

        for r in self.curation_result.training_collection.curve_records:
            fam_id = family_map.get(r.dataset_id)
            if fam_id in train_fams:
                train_recs.append(r)
            elif fam_id in val_fams:
                val_recs.append(r)
            elif fam_id in test_fams:
                test_recs.append(r)

        # Fit preprocessor STRICTLY on training records only!
        preprocessor = MLPreprocessor()
        preprocessor.fit(train_recs)

        return {
            "train": CurvePointDataset(train_recs, preprocessor),
            "val": CurvePointDataset(val_recs, preprocessor),
            "test": CurvePointDataset(test_recs, preprocessor),
        }

    def prepare_regime_b_datasets(self) -> Dict[str, CurvePointDataset]:
        """Prepare train and test CurvePointDatasets for REGIME_B_CROSS_MATERIAL."""
        train_fams = set(self.curation_result.regime_b_train_family_ids)
        test_fams = set(self.curation_result.regime_b_test_family_ids)

        family_map = {e.dataset_id: e.experiment_family_id for e in self.curation_result.manifest_entries}

        train_recs = []
        test_recs = []

        for r in self.curation_result.training_collection.curve_records:
            fam_id = family_map.get(r.dataset_id)
            if fam_id in train_fams:
                train_recs.append(r)
            elif fam_id in test_fams:
                test_recs.append(r)

        # Fit preprocessor STRICTLY on training records only!
        preprocessor = MLPreprocessor()
        preprocessor.fit(train_recs)

        return {
            "train": CurvePointDataset(train_recs, preprocessor),
            "test": CurvePointDataset(test_recs, preprocessor),
        }

    def run_baseline_evaluation(self, regime: SplitRegime = SplitRegime.REGIME_A_IN_DOMAIN) -> Dict[str, Any]:
        """Benchmark baseline models on specified split regime."""
        if regime == SplitRegime.REGIME_A_IN_DOMAIN:
            datasets = self.prepare_regime_a_datasets()
            trainer = SolarModelTrainer(self.config)
            trainer.train_baseline_models(datasets["train"])
            val_metrics = trainer.evaluate_baseline_models(datasets["val"])
            test_metrics = trainer.evaluate_baseline_models(datasets["test"])
            return {
                "regime": "REGIME_A_IN_DOMAIN",
                "val_metrics": {k: v.to_dict() for k, v in val_metrics.items()},
                "test_metrics": {k: v.to_dict() for k, v in test_metrics.items()},
            }
        else:
            datasets = self.prepare_regime_b_datasets()
            trainer = SolarModelTrainer(self.config)
            trainer.train_baseline_models(datasets["train"])
            test_metrics = trainer.evaluate_baseline_models(datasets["test"])
            return {
                "regime": "REGIME_B_CROSS_MATERIAL",
                "test_metrics": {k: v.to_dict() for k, v in test_metrics.items()},
            }
