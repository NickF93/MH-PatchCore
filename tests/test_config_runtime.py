from pathlib import Path

from mhpc.core.plugins.default_bundle import build_default_plugin_bundle
from mhpc.eval.config import load_run_config


def test_all_mvtec_configs_load_and_select_available_plugins() -> None:
    config_paths = sorted(Path("configs/mvtec").glob("*/*.yaml"))
    assert config_paths

    for config_path in config_paths:
        cfg = load_run_config(config_path)
        assert cfg.runtime.device == "cpu"
        bundle = build_default_plugin_bundle(cfg.plugins.as_selection_map())
        assert bundle.dataloader_plugin is not None
        assert bundle.model_plugin_bundle.scoring_plugin is not None
