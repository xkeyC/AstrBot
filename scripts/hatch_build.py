"""
Custom Hatchling build hook.

Standard wheel builds automatically build the dashboard when compatible bundled
assets are not already present, including `uv tool install git+...`. Editable
installs are skipped unless ASTRBOT_BUILD_DASHBOARD=1 is set.

Set ASTRBOT_BUILD_DASHBOARD=1 to force a rebuild, or 0 to disable it.

When enabled, this hook:
1. Runs the dashboard package-manager build inside the `dashboard/` directory.
2. Copies the resulting `dashboard/dist/` tree into
   `astrbot/dashboard/dist/` so the static assets are shipped
   inside the Python wheel.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    @staticmethod
    def _run(command: list[str], cwd: Path) -> None:
        print(f"[hatch_build] Running: {' '.join(command)}")
        subprocess.run(command, cwd=cwd, check=True)

    @staticmethod
    def _resolve_command(command: str) -> str | None:
        if os.name == "nt":
            return shutil.which(f"{command}.cmd") or shutil.which(command)
        return shutil.which(command)

    @staticmethod
    def _should_build_dashboard(build_version: str) -> bool:
        env_value = os.environ.get("ASTRBOT_BUILD_DASHBOARD", "").strip().lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
        if env_value in {"0", "false", "no", "off"}:
            return False
        return build_version != "editable"

    def initialize(self, version: str, build_data: dict) -> None:
        del build_data

        build_setting = os.environ.get("ASTRBOT_BUILD_DASHBOARD", "").strip().lower()
        force_build = build_setting in {"1", "true", "yes", "on"}
        if self.target_name != "wheel" or not self._should_build_dashboard(version):
            return

        root = Path(self.root)
        dashboard_src = root / "dashboard"
        dist_src = dashboard_src / "dist"
        dist_target = root / "astrbot" / "dashboard" / "dist"
        package_init = root / "astrbot" / "__init__.py"
        version_match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            package_init.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if version_match is None:
            raise RuntimeError("Unable to read the AstrBot release version.")
        expected_version = f"v{version_match.group(1)}"
        target_version_file = dist_target / "assets" / "version"

        if (
            not force_build
            and (dist_target / "index.html").is_file()
            and target_version_file.is_file()
            and target_version_file.read_text(encoding="utf-8").strip()
            == expected_version
        ):
            print(
                "[hatch_build] Compatible bundled dashboard already exists; "
                "skipping npm."
            )
            return

        if not dashboard_src.exists():
            raise RuntimeError(
                "Dashboard source and compatible bundled assets are both missing."
            )

        uses_pnpm = (dashboard_src / "pnpm-lock.yaml").exists()
        pnpm_executable = self._resolve_command("pnpm")
        if uses_pnpm:
            if pnpm_executable:
                pnpm_command = [pnpm_executable]
            else:
                npx_executable = self._resolve_command("npx")
                if npx_executable:
                    pnpm_command = [npx_executable, "--yes", "pnpm@9"]
                else:
                    raise RuntimeError(
                        "pnpm is required to build dashboard, and neither pnpm nor "
                        "npx is available. Install Node.js/npm first."
                    )
        else:
            npm_executable = self._resolve_command("npm")
            if npm_executable is None:
                raise RuntimeError(
                    "npm is required to build dashboard assets for the Python wheel."
                )
            pnpm_command = []

        package_manager = " ".join(pnpm_command) if uses_pnpm else "npm"
        install_command = (
            [*pnpm_command, "install", "--frozen-lockfile"]
            if uses_pnpm
            else [npm_executable, "install"]
        )
        build_command = (
            [*pnpm_command, "run", "build-local"]
            if uses_pnpm
            else [npm_executable, "run", "build-local"]
        )

        # ── Sync Node dependencies before building ───────────────────────────
        print(
            f"[hatch_build] Installing dashboard Node dependencies with {package_manager}..."
        )
        self._run(install_command, cwd=dashboard_src)

        # ── Build the Vue/Vite dashboard ──────────────────────────────────────
        print(f"[hatch_build] Building Vue dashboard ({' '.join(build_command)})...")
        self._run(build_command, cwd=dashboard_src)

        if not dist_src.exists():
            raise RuntimeError("dashboard/dist was not created by npm run build.")

        version_file = dist_src / "assets" / "version"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        version_file.write_text(expected_version, encoding="utf-8")

        # ── Copy into the Python package tree ────────────────────────────────
        if dist_target.exists():
            shutil.rmtree(dist_target)
        shutil.copytree(dist_src, dist_target)
        print(f"[hatch_build] Dashboard dist copied → {dist_target.relative_to(root)}")
