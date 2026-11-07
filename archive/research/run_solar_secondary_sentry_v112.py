"""Freeze and launch only the sentry with the full v84 cache source pins."""

import sys

import run_solar_secondary_v112 as supervisor


def main():
    if sys.argv[1:3] not in (["freeze", "sentry"], ["launch", "sentry"]):
        raise SystemExit("Use freeze sentry or launch sentry with the publication commit.")
    supervisor.SENTRY_SOURCES += (
        "run_solar_secondary_sentry_v112.py",
        "solar_acquisition_cache_v84.py",
        "solar_response_cache_v84.py",
        "solar_pose_proposals_v73.py",
        "reports/solar_learning_contract_v79.json",
        "reports/solar_reference_contract_v80.json",
    )
    supervisor.main()


if __name__ == "__main__":
    main()
