"""Execute every example or notebook and write a reproducible QA manifest.

Uses the current interpreter and fresh subprocesses/kernels. Notebooks are
executed in memory, never overwritten. --offline requires cached checkpoints.
Install the documented extras before treating failures/skips as SDK evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebooks', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
    if args.offline:
        env['HF_HUB_OFFLINE'] = '1'
    files = (root / ('notebooks' if args.notebooks else 'examples')).glob(
        '*.ipynb' if args.notebooks else '*.py')
    results = []
    for path in sorted(files):
        start = time.monotonic()
        status, output = 'passed', ''
        try:
            if args.notebooks:
                import nbformat
                from nbclient import NotebookClient
                from jupyter_client.manager import KernelManager
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                manager = KernelManager(kernel_name='python3')
                manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                            '-f', '{connection_file}']
                try:
                    NotebookClient(notebook, timeout=args.timeout, km=manager,
                                   resources={'metadata': {'path': str(root)}}).execute(env=env)
                finally:
                    if manager.has_kernel:
                        manager.shutdown_kernel(now=True)
            else:
                result = subprocess.run([sys.executable, str(path)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=args.timeout)
                status = 'passed' if result.returncode == 0 else 'failed'
                output = result.stdout + result.stderr
        except Exception as exc:
            status, output = type(exc).__name__, str(exc)
        (args.output / (path.stem + '.log')).write_text(output)
        results.append(dict(file=path.name, status=status,
                            seconds=round(time.monotonic()-start, 2)))
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(path.name, status, flush=True)
    return int(any(row['status'] != 'passed' for row in results))


if __name__ == '__main__':
    sys.exit(main())
