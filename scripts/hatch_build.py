"""
Custom Hatchling build hook.

Only runs when the environment variable ASTRBOT_BUILD_DASHBOARD=1 is set,
so that `uv sync` / editable installs are never affected.

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

    def initialize(self, version: str, build_data: dict) -> None:
        # Only run when explicitly requested (e.g. during CI / release builds).
        # This prevents `uv sync` / editable installs from triggering npm.
        if os.environ.get("ASTRBOT_BUILD_DASHBOARD", "").strip() != "1":
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
        if uses_pnpm and not self._has_command("pnpm"):
            if not self._has_command("npm"):
                raise RuntimeError(
                    "pnpm is required to build dashboard, and npm is not available "
                    "to install it. Install Node.js/npm first."
                )
            self._run(["npm", "install", "-g", "pnpm@9"], cwd=dashboard_src)

        package_manager = "pnpm" if uses_pnpm else "npm"
        install_command = (
            ["pnpm", "install", "--frozen-lockfile"]
            if uses_pnpm
            else ["npm", "install"]
        )
        build_command = (
            ["pnpm", "run", "build-prod"] if uses_pnpm else ["npm", "run", "build-prod"]
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
        version_file.write_text(version, encoding="utf-8")
        print(f"[hatch_build] Dashboard dist copied → {dist_target.relative_to(root)}")
