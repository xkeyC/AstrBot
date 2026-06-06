"""
Custom Hatchling build hook.

Runs by default for wheel builds, including `uv tool install git+...`.
Editable installs are skipped unless ASTRBOT_BUILD_DASHBOARD=1 is set,
so that `uv sync` is not forced to run Node.js tooling.

Usage:
    ASTRBOT_BUILD_DASHBOARD=1 uv build

When enabled, this hook:
1. Runs the dashboard package-manager build inside the `dashboard/` directory.
2. Copies the resulting `dashboard/dist/` tree into
   `astrbot/dashboard/dist/` so the static assets are shipped
   inside the Python wheel.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import tomllib
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    @staticmethod
    def _run(command: list[str], cwd: Path) -> None:
        print(f"[hatch_build] Running: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=True)

    @staticmethod
    def _has_command(command: str) -> bool:
        return shutil.which(command) is not None

    @staticmethod
    def _should_build_dashboard(build_version: str) -> bool:
        env_value = os.environ.get("ASTRBOT_BUILD_DASHBOARD", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        if env_value in {"0", "false", "no", "off"}:
            return False
        return build_version != "editable"

    @staticmethod
    def _read_project_version(root: Path) -> str:
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project_version = pyproject.get("project", {}).get("version")
        if not isinstance(project_version, str) or not project_version.strip():
            raise RuntimeError("Unable to read project.version from pyproject.toml")
        return project_version.strip()

    def initialize(self, version: str, build_data: dict) -> None:
        del build_data

        if not self._should_build_dashboard(version):
            return

        root = Path(self.root)
        dashboard_src = root / "dashboard"
        dist_src = dashboard_src / "dist"
        dist_target = root / "astrbot" / "dashboard" / "dist"

        if not dashboard_src.exists():
            print(
                "[hatch_build] 'dashboard/' directory not found – skipping dashboard build.",
                file=sys.stderr,
            )
            return

        uses_pnpm = (dashboard_src / "pnpm-lock.yaml").exists()
        pnpm_command = ["pnpm"]
        if uses_pnpm and not self._has_command("pnpm"):
            if self._has_command("npx"):
                pnpm_command = ["npx", "--yes", "pnpm@9"]
            else:
                raise RuntimeError(
                    "pnpm is required to build dashboard, and neither pnpm nor npx "
                    "is available. Install Node.js/npm first."
                )

        package_manager = " ".join(pnpm_command) if uses_pnpm else "npm"
        install_command = (
            [*pnpm_command, "install", "--frozen-lockfile"]
            if uses_pnpm
            else ["npm", "install"]
        )
        build_command = (
            [*pnpm_command, "run", "build-local"]
            if uses_pnpm
            else ["npm", "run", "build-local"]
        )

        # ── Install Node dependencies if node_modules is absent ─────────────
        if not (dashboard_src / "node_modules").exists():
            print(
                f"[hatch_build] Installing dashboard Node dependencies with {package_manager}..."
            )
            self._run(install_command, cwd=dashboard_src)

        # ── Build the Vue/Vite dashboard ──────────────────────────────────────
        print(f"[hatch_build] Building Vue dashboard ({' '.join(build_command)})...")
        self._run(build_command, cwd=dashboard_src)

        if not dist_src.exists():
            print(
                "[hatch_build] dashboard/dist not found after build – skipping copy.",
                file=sys.stderr,
            )
            return

        # ── Copy into the Python package tree ────────────────────────────────
        if dist_target.exists():
            shutil.rmtree(dist_target)
        shutil.copytree(dist_src, dist_target)
        version_file = dist_target / "assets" / "version"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(f"v{self._read_project_version(root).lstrip('v')}", encoding="utf-8")
        print(f"[hatch_build] Dashboard dist copied → {dist_target.relative_to(root)}")
