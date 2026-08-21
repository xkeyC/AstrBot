"""
Custom Hatchling build hook.

Standard wheel builds automatically build the dashboard when compatible bundled
assets are not already present. Editable installs remain unaffected.

Set ASTRBOT_BUILD_DASHBOARD=1 to force a rebuild, or 0 to disable it.

When enabled, this hook:
1. Runs `npm run build` inside the `dashboard/` directory.
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

    def initialize(self, version: str, build_data: dict) -> None:
        build_setting = os.environ.get("ASTRBOT_BUILD_DASHBOARD", "").strip()
        force_build = build_setting == "1"
        if build_setting.lower() in {"0", "false", "no", "off"}:
            return
        if self.target_name != "wheel" or (version == "editable" and not force_build):
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

        npm_command = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if npm_command is None:
            raise RuntimeError(
                "npm is required to build dashboard assets for the Python wheel."
            )

        # ── Install Node dependencies if node_modules is absent ─────────────
        if not (dashboard_src / "node_modules").exists():
            print("[hatch_build] Installing dashboard Node dependencies...")
            subprocess.run(
                [npm_command, "install"],
                cwd=dashboard_src,
                check=True,
            )

        # ── Build the Vue/Vite dashboard ──────────────────────────────────────
        print("[hatch_build] Building Vue dashboard (npm run build)...")
        subprocess.run(
            [npm_command, "run", "build"],
            cwd=dashboard_src,
            check=True,
        )

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
