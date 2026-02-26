from tileflow.settings import SettingsStore, TileFlowSettings


def test_settings_load_recovers_from_corrupt_json(tmp_path) -> None:
    store = SettingsStore(home=tmp_path)
    store.path.write_text("{bad", encoding="utf-8")
    loaded = store.load()
    assert isinstance(loaded, TileFlowSettings)
    assert loaded.ktransformers_path is None
    assert store.path.exists()

