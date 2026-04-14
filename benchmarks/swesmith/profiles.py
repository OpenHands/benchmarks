"""
Custom repo profiles for SWE-Smith evaluation.

Profiles defined here are auto-registered with the swesmith global registry
on import. To add a new repo, define a dataclass inheriting from the
appropriate base (GoProfile, PythonProfile, etc.) and it will be picked up
automatically.

Usage in eval_infer.py:
    import benchmarks.swesmith.profiles  # noqa: F401
"""

from dataclasses import dataclass

from swesmith.profiles import registry  # triggers __init__.py → registers all languages
from swesmith.profiles.base import RepoProfile
from swesmith.profiles.golang import GoProfile
from swesmith.profiles.python import PythonProfile


# ---------------------------------------------------------------------------
# Monkey-patch: use image_name from the task instance dataset
#
# swesmith's RepoProfile.image_name is a @property that computes the Docker
# image name from profile fields. However, the computed name can differ from
# the actual image name stored in the task instance dataset (which was set at
# image build time and is the source of truth).
#
# Instead of recomputing the name, we patch the lookup to use the value
# directly from the task instance:
#
# 1. Patch registry.get_from_inst() to stash instance["image_name"] keyed
#    by repo_name when the harness resolves a profile from an instance.
# 2. Patch RepoProfile.image_name to return the stashed value when available,
#    falling back to the original computation otherwise.
# ---------------------------------------------------------------------------
_instance_image_names: dict[str, str] = {}

_original_get_from_inst = registry.get_from_inst


def _patched_get_from_inst(instance):
    rp = _original_get_from_inst(instance)
    if "image_name" in instance:
        _instance_image_names[rp.repo_name] = instance["image_name"]
    return rp


registry.get_from_inst = _patched_get_from_inst

_original_image_name_fget = RepoProfile.image_name.fget
assert _original_image_name_fget is not None
_image_name_getter = _original_image_name_fget


@property
def _patched_image_name(self):
    override = _instance_image_names.get(self.repo_name)
    if override is not None:
        return override
    return _image_name_getter(self)


RepoProfile.image_name = _patched_image_name  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Custom profiles — add your repo profiles below.
# ---------------------------------------------------------------------------


@dataclass
class SecretGoProject2c88df8f(GoProfile):
    owner: str = "studentkaramuk"
    repo: str = "secret-go-project"
    commit: str = "2c88df8f24627306470fb88dd4d89f11cee3408d"
    org_gh: str = "studentkaramuk-swesmith"


@dataclass
class BookSummaryf26f9b51(PythonProfile):
    owner: str = "reisepass"
    repo: str = "book_chapter_detection_and_summarization"
    commit: str = "f26f9b510449cd0bc7aacc2f504d793aed43bc96"
    org_gh: str = "code-peerbench"
    test_cmd: str = (
        "source /opt/miniconda3/bin/activate; "
        "conda activate testbed; "
        "ELEVENLABS_API_KEY=dummy "
        "pytest tests/ --disable-warnings --color=no --tb=no --verbose"
    )


@dataclass
class Httpxae1b9f66(PythonProfile):
    owner: str = "encode"
    repo: str = "httpx"
    commit: str = "ae1b9f66238f75ced3ced5e4485408435de10768"
    org_gh: str = "studentkaramuk-swesmith"


# ---- Auto-register all profiles defined above ----
_BASE_CLASSES = {RepoProfile, GoProfile, PythonProfile}

for _name, _obj in list(globals().items()):
    if (
        isinstance(_obj, type)
        and issubclass(_obj, RepoProfile)
        and _obj not in _BASE_CLASSES
    ):
        registry.register_profile(_obj)
