from pathlib import Path


def test_training_configuration_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "tiny_stories.yaml").is_file()


def test_model_module_imports():
    import minigpt.model  # noqa: F401
