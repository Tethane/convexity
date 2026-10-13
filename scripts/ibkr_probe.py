"""Thin CLI for Convexity's read-only paper account API."""

import argparse
import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

# Allow the documented source-checkout command without an editable installation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from convexity.brokers.ibkr import PaperAccountConfig, PaperAccountError, read_paper_account


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redact", action="store_true", help="Mask account identity for a shareable output sample (financial values remain visible).")
    args = parser.parse_args()
    # The client can log unfiltered account objects. Our errors provide diagnostics
    # without leaking another account's data, including when redaction is enabled.
    logging.getLogger("ib_async").disabled = True
    for name in ("ib_async.ib", "ib_async.client", "ib_async.wrapper"):
        logging.getLogger(name).disabled = True
    try:
        config = PaperAccountConfig.from_env()
        snapshot = asyncio.run(read_paper_account(config))
    except (OSError, ValueError, PaperAccountError) as exc:
        print(f"Paper account probe failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Paper account probe interrupted; disconnected.", file=sys.stderr)
        return 130
    result = asdict(snapshot)
    if args.redact:
        result["account"] = "DU[REDACTED]"
    result["connection"] = "verified; disconnected cleanly"
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
