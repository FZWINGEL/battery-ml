"""Tests for preprocessing pipelines."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSummarySetPipeline:
    """Tests for SummarySetPipeline."""

    def test_fit_transform(self, sample_summary_df):
        """Test fit_transform produces valid samples."""
        from src.pipelines.summary_set import SummarySetPipeline

        pipeline = SummarySetPipeline(include_arrhenius=True, normalize=True)
        samples = pipeline.fit_transform({"df": sample_summary_df})

        assert len(samples) == len(sample_summary_df)
        assert samples[0].x is not None
        assert samples[0].y is not None
        assert "cell_id" in samples[0].meta

    def test_feature_names(self, sample_summary_df):
        """Test feature names are correct."""
        from src.pipelines.summary_set import SummarySetPipeline

        pipeline = SummarySetPipeline(include_arrhenius=True)
        pipeline.fit({"df": sample_summary_df})

        names = pipeline.get_feature_names()
        assert "temp_K" in names
        assert "arrhenius" in names

    def test_normalization(self, sample_summary_df):
        """Test normalization is applied."""
        from src.pipelines.summary_set import SummarySetPipeline

        pipeline = SummarySetPipeline(normalize=True)
        samples = pipeline.fit_transform({"df": sample_summary_df})

        # Normalized features should have reasonable range
        x_values = np.vstack([s.x for s in samples])
        assert x_values.std() < 10  # Roughly normalized


class TestICAPeaksPipeline:
    """Tests for ICAPeaksPipeline."""

    def test_compute_ica(self, sample_voltage_curve):
        """Test ICA computation."""
        from src.pipelines.ica_peaks import ICAPeaksPipeline

        voltage, capacity = sample_voltage_curve
        pipeline = ICAPeaksPipeline(sg_window=21, num_peaks=3, use_cache=False)

        V_mid, dQdV = pipeline.compute_ica(voltage, capacity)

        assert len(V_mid) > 0
        assert len(dQdV) == len(V_mid)

    def test_extract_peak_features(self, sample_voltage_curve):
        """Test peak feature extraction."""
        from src.pipelines.ica_peaks import ICAPeaksPipeline

        voltage, capacity = sample_voltage_curve
        pipeline = ICAPeaksPipeline(sg_window=21, num_peaks=3, use_cache=False)

        V_mid, dQdV = pipeline.compute_ica(voltage, capacity)
        features = pipeline.extract_peak_features(V_mid, dQdV)

        assert len(features) == len(pipeline.get_feature_names())

    def test_fit_transform(self, sample_curves_with_meta):
        """Test full pipeline."""
        from src.pipelines.ica_peaks import ICAPeaksPipeline

        curves, targets = sample_curves_with_meta
        pipeline = ICAPeaksPipeline(use_cache=False, normalize=True)

        samples = pipeline.fit_transform({"curves": curves, "targets": targets})

        assert len(samples) == len(curves)


class TestLatentODESequencePipeline:
    """Tests for LatentODESequencePipeline."""

    def test_sequence_creation(self, sample_summary_df):
        """Test sequence sample creation."""
        from src.pipelines.latent_ode_seq import LatentODESequencePipeline

        pipeline = LatentODESequencePipeline(time_unit="days")
        samples = pipeline.fit_transform({"df": sample_summary_df})

        # Should have one sample per cell
        assert len(samples) == sample_summary_df["cell_id"].nunique()

        # Check time vector exists
        assert samples[0].t is not None
        assert samples[0].t[0] == 0  # Time starts at 0


class TestPipelineRegistry:
    """Tests for PipelineRegistry."""

    def test_list_available(self):
        """Test listing available pipelines."""
        from src.pipelines.registry import PipelineRegistry

        # Import pipelines to register them
        from src.pipelines import summary_set, ica_peaks, latent_ode_seq  # noqa: F401

        available = PipelineRegistry.list_available()

        assert "summary_set" in available
        assert "ica_peaks" in available
        assert "latent_ode_seq" in available

    def test_get_pipeline(self):
        """Test getting pipeline by name."""
        from src.pipelines.registry import PipelineRegistry
        from src.pipelines import summary_set  # noqa: F401

        pipeline = PipelineRegistry.get("summary_set", include_arrhenius=True)

        assert pipeline.include_arrhenius
