"""Basic import and sanity tests for ProtoGlue."""

import pytest


def test_import():
    """Test that protoglue can be imported."""
    import protoglue
    assert hasattr(protoglue, "__version__")


def test_version():
    """Test that version string is valid."""
    import protoglue
    parts = protoglue.__version__.split(".")
    assert len(parts) == 3


def test_import_model():
    """Test that ProtoGlue model can be imported."""
    from protoglue.models import ProtoGlue
    assert ProtoGlue is not None


def test_import_data():
    """Test that data modules can be imported."""
    from protoglue.data import load_and_preprocess, build_graphs
    assert callable(load_and_preprocess)
    assert callable(build_graphs)


def test_import_training():
    """Test that training modules can be imported."""
    from protoglue.training import run_stage, scan_select_k
    assert callable(run_stage)
    assert callable(scan_select_k)


def test_import_losses():
    """Test that loss modules can be imported."""
    from protoglue.losses import total_loss, stable_vicreg_loss
    assert callable(total_loss)
    assert callable(stable_vicreg_loss)


def test_import_api():
    """Test that high-level API can be imported."""
    from protoglue import run_pipeline
    assert callable(run_pipeline)


def test_model_creation():
    """Test that a ProtoGlue model can be instantiated."""
    import torch
    from protoglue.models import ProtoGlue

    model = ProtoGlue(in1=50, in2=30, n_scales=3)
    assert model is not None

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_dec_head():
    """Test that DECHead works correctly."""
    import torch
    from protoglue.models.decoders import DECHead

    dec = DECHead(n_clusters=5, d=64)
    z = torch.randn(10, 64)
    q = dec.soft_assign(z)
    assert q.shape == (10, 5)
    assert torch.allclose(q.sum(dim=1), torch.ones(10), atol=1e-5)

    p = DECHead.target_distribution(q)
    assert p.shape == (10, 5)


def test_fix_seed():
    """Test that fix_seed runs without error."""
    from protoglue.data.utils import fix_seed
    fix_seed(42)
