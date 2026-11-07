"""One maintained entry point for evidence, validation and experimental matching."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .artifacts import audit_archive, write_json_exclusive
from .evidence import format_table, load_evidence, summary
from .paths import repository_root

POLICIES = ("single_uniform", "beam_uniform", "beam_geometry", "beam_active")


def budget(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an integer field budget.") from error
    if not 0 <= number <= 128:
        raise argparse.ArgumentTypeError("Field budget must be between 0 and 128.")
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="lunar-match", description=__doc__)
    result.add_argument("--version", action="version", version=__version__)
    commands = result.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Show the complete measured pilot; no matching.")
    status.add_argument("--json", action="store_true")
    commands.add_parser("audit", help="Verify every archived public file against its original bytes.")
    for name in ("smoke", "replay"):
        command = commands.add_parser(name, help=("Generated software control; no lunar images."
            if name == "smoke" else "Replay saved geometry and accounting; no matcher."))
        command.add_argument("--output", type=Path)
    inputs = argparse.ArgumentParser(add_help=False)
    inputs.add_argument("--manifest", type=Path, required=True)
    inputs.add_argument("--case-id", required=True)
    inputs.add_argument("--input-root", type=Path, help="Base for manifest paths; defaults to the manifest directory.")
    validate = commands.add_parser("validate", parents=[inputs], help="Verify a native input case without running a matcher.")
    validate.add_argument("--output", type=Path)
    localize = commands.add_parser("localize", parents=[inputs], help="Experimental native matching; public pose remains null.")
    localize.add_argument("--output", type=Path, required=True)
    localize.add_argument("--field-budget", type=budget, default=128)
    localize.add_argument("--policies", nargs="+", choices=POLICIES, default=["beam_active"])
    localize.add_argument("--research-output", action="store_true")
    return result


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "status":
            evidence = load_evidence()
            print(json.dumps(summary(evidence), indent=2) if arguments.json else format_table(evidence))
            return 0
        if arguments.command == "audit":
            result = audit_archive(repository_root())
        else:
            if arguments.output is not None and arguments.output.exists():
                raise FileExistsError("The output already exists; choose a new result path.")
            from .backend import run_backend
            if arguments.command in {"smoke", "replay"}:
                result = run_backend(arguments.command)
                if result.get("passed") is not True:
                    raise RuntimeError("The control or replay did not pass.")
            else:
                from .validation import prepare_case
                manifest = arguments.manifest.resolve()
                base = arguments.input_root.resolve() if arguments.input_root else manifest.parent
                if arguments.command == "localize" and len(set(arguments.policies)) != len(arguments.policies):
                    raise ValueError("Choose distinct acquisition policies.")
                case = prepare_case(manifest, arguments.case_id, base)
                if arguments.command == "validate":
                    result = {"passed": True, "case_id": case["id"], "reference_lights": len(case["catalogue"]),
                              "query_truth_used": False, "matching_run": False}
                else:
                    result = run_backend("localize", case=case, policies=arguments.policies,
                                         field_budget=arguments.field_budget,
                                         research_output=arguments.research_output)
                    if result.get("pose") is not None or result.get("acceptance_status") != "uncalibrated":
                        raise RuntimeError("An uncalibrated public result cannot expose an accepted pose.")
                    result["accepted"] = False
            if arguments.output is not None:
                write_json_exclusive(arguments.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        print(f"lunar-match: {error}", file=sys.stderr)
        return 2
