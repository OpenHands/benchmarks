"""
Runtime patches for third-party dependencies.

This adjusts the swebench harness install commands to remain compatible with
pip>=24 (which removed the --no-use-pep517 flag) by removing the flag and
pinning pip<24 for affected repos.
"""

from __future__ import annotations


def _patch_swebench_install_flags() -> None:
    try:
        from swebench.harness.constants import python as py_consts  # type: ignore
    except Exception:
        # swebench isn't installed in this environment
        return

    specs_map = getattr(py_consts, "MAP_REPO_VERSION_TO_SPECS_PY", {})
    patched_repos: list[str] = []

    for repo, versions in specs_map.items():
        repo_updated = False
        for spec in versions.values():
            install = spec.get("install")
            if install and "--no-use-pep517" in install:
                # Remove deprecated flag and ensure an older pip is available
                spec["install"] = install.replace("--no-use-pep517", "").strip()
                pip_packages = spec.setdefault("pip_packages", [])
                if not any(pkg.lower().startswith("pip") for pkg in pip_packages):
                    pip_packages.insert(0, "pip<24")
                repo_updated = True
        if repo_updated:
            patched_repos.append(repo)

    if not patched_repos:
        return

    try:
        from swebench.harness import constants as root_consts  # type: ignore
    except Exception:
        return

    for repo in patched_repos:
        if repo in root_consts.MAP_REPO_VERSION_TO_SPECS:
            root_consts.MAP_REPO_VERSION_TO_SPECS[repo] = specs_map[repo]


_patch_swebench_install_flags()
