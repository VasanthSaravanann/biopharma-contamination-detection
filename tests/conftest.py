import importlib
import pytest


def pytest_collection_modifyitems(config, items):
    # If torch is not importable, skip heavy neural-network tests
    if importlib.util.find_spec('torch') is None:
        skip_marker = pytest.mark.skip(reason='torch not installed; skipping heavy NN tests')
        for item in items:
            # heuristics: skip tests that import mh_ddpm or rely on deep models
            if 'test_enhancements.py' in str(item.fspath) or 'mh_ddpm' in str(item.fspath):
                item.add_marker(skip_marker)
