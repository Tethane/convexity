"""Prompt locally for Alpaca keys and store outside the source tree, owner-only."""
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Run this script in a local interactive terminal for hidden key input.")
    directory = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "convexity"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    key = getpass.getpass("Alpaca API key (hidden): ").strip()
    secret = getpass.getpass("Alpaca secret key (hidden): ").strip()
    if not key or not secret or any(c.isspace() for c in key + secret):
        raise SystemExit("Both keys must be nonempty and contain no whitespace. Nothing saved.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            json.dump(dict(api_key=key, secret_key=secret), stream)
            stream.write("\n")
        temporary.replace(directory / "alpaca.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved credentials to {directory / 'alpaca.json'} (permissions 600). Keys were not printed.")


if __name__ == "__main__":
    main()
