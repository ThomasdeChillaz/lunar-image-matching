"""Compatibility with the original script names, without changing frozen source."""
from pathlib import Path
import subprocess
import sys

from .paths import research_root

PATH_OPTIONS = {"--reference", "--query", "--reference-geometry", "--query-intrinsics", "--config",
                "--checkpoint", "--model-dir", "--protocol", "--solar-lookup-protocol", "--output"}


def legacy_main(script: str, argv=None) -> int:
    if script not in {"match_terrain.py", "localize_terrain.py"}:
        raise ValueError("Unsupported compatibility command.")
    original = Path.cwd()
    args = list(sys.argv[1:] if argv is None else argv)
    for index, item in enumerate(args):
        option, equals, value = item.partition("=")
        if option not in PATH_OPTIONS:
            continue
        at = index if equals else index + 1
        if not equals and at >= len(args):
            continue  # The original argument parser reports the missing value.
        value = value if equals else args[at]
        if not value or value.startswith("--"):
            continue
        path = Path(value)
        if not path.is_absolute():
            here = original / path
            archived = research_root() / path
            path = here if option == "--output" or here.exists() or not archived.exists() else archived
        args[at] = option + "=" + str(path) if equals else str(path)
    return subprocess.run([sys.executable, str(research_root() / script), *args], cwd=research_root()).returncode
