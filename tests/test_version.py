"""Tests for benchmarks.utils.version IMAGE_TAG_PREFIX resolution."""

import importlib
import os
from unittest.mock import patch

import pytest


def _reload_version(**env_overrides):
    """Reload version module with custom environment variables."""
    import benchmarks.utils.version as version_mod

    with patch.dict(os.environ, env_overrides, clear=False):
        # Remove env vars not in overrides so we test clean state
        for key in ("IMAGE_TAG_PREFIX", "SDK_SHORT_SHA"):
            if key not in env_overrides:
                os.environ.pop(key, None)
        importlib.reload(version_mod)
    return version_mod


class TestImageTagPrefix:
    def teardown_method(self):
        """Restore version module to default state after each test."""
        import benchmarks.utils.version as version_mod

        for key in ("IMAGE_TAG_PREFIX", "SDK_SHORT_SHA"):
            os.environ.pop(key, None)
        importlib.reload(version_mod)

    def test_default_uses_sdk_short_sha(self):
        """When no env vars are set, IMAGE_TAG_PREFIX defaults to SDK_SHORT_SHA."""
        mod = _reload_version()
        assert mod.IMAGE_TAG_PREFIX == mod.SDK_SHORT_SHA

    def test_image_tag_prefix_env_override(self):
        """IMAGE_TAG_PREFIX env var overrides the default."""
        mod = _reload_version(IMAGE_TAG_PREFIX="custom-tag")
        assert mod.IMAGE_TAG_PREFIX == "custom-tag"

    def test_deprecated_sdk_short_sha_env_fallback(self):
        """SDK_SHORT_SHA env var is honored with a deprecation warning."""
        with pytest.warns(DeprecationWarning, match="SDK_SHORT_SHA"):
            mod = _reload_version(SDK_SHORT_SHA="legacy-tag")
        assert mod.IMAGE_TAG_PREFIX == "legacy-tag"

    def test_image_tag_prefix_takes_precedence_over_sdk_short_sha(self):
        """IMAGE_TAG_PREFIX env var wins over deprecated SDK_SHORT_SHA."""
        mod = _reload_version(IMAGE_TAG_PREFIX="new-tag", SDK_SHORT_SHA="old-tag")
        assert mod.IMAGE_TAG_PREFIX == "new-tag"


class TestPhasedImageTagPrefix:
    def teardown_method(self):
        """Restore version module to default state after each test."""
        import benchmarks.utils.version as version_mod

        for key in ("IMAGE_TAG_PREFIX", "SDK_SHORT_SHA"):
            os.environ.pop(key, None)
        importlib.reload(version_mod)

    def test_default_includes_sdk_sha_and_content_hash(self):
        """When no env vars are set, phased prefix is SDK_SHORT_SHA + content hash."""
        mod = _reload_version()
        prefix = mod.get_phased_image_tag_prefix()
        assert prefix.startswith(mod.SDK_SHORT_SHA + "-")
        # The suffix is the 7-char Dockerfile content hash
        content_hash = prefix[len(mod.SDK_SHORT_SHA) + 1 :]
        assert len(content_hash) == 7
        assert content_hash.isalnum()

    def test_env_override(self):
        """IMAGE_TAG_PREFIX env var overrides phased prefix too."""
        mod = _reload_version()
        with patch.dict(os.environ, {"IMAGE_TAG_PREFIX": "custom-tag"}):
            assert mod.get_phased_image_tag_prefix() == "custom-tag"

    def test_deprecated_sdk_short_sha_env_fallback(self):
        """SDK_SHORT_SHA env var is honored for phased prefix."""
        with pytest.warns(DeprecationWarning, match="SDK_SHORT_SHA"):
            mod = _reload_version(SDK_SHORT_SHA="legacy-tag")
        assert mod.get_phased_image_tag_prefix() == "legacy-tag"
