"""Re-plot saved results without re-running training.

Run from the reinforcement_learning/ folder, after main.py has saved results:

    python -m homework1.plot_saved                          # defaults
    python -m homework1.plot_saved --window 500             # smoother curves
    python -m homework1.plot_saved --layout columns         # side by side
    python -m homework1.plot_saved --save results/hw1.png   # save the figure for the report

Compare runs from different files - repeat --file, give each a --label, and use
--method to keep just one update method from each file:

    python -m homework1.plot_saved --method "constant step size" \\
        --file results/hw1_eps_decay.npz  --label "ε-greedy with ε decay" \\
        --file results/hw1_temp_decay.npz --label "softmax with τ decay"
"""
import argparse
import os
import re

# from homework1.plotting import plot_bandit_results
from homework1.plotting_spiral_randomwalk import plot_bandit_results
from homework1.results_io import load_results


def _select_method(results, method, path):
    """Return the single label in `results` that starts with `method` (case-insensitive)."""
    matches = [label for label in results if label.lower().startswith(method.lower())]
    if len(matches) != 1:
        raise SystemExit(f"--method {method!r} matched {len(matches)} methods in {path}. "
                         f"Available: {list(results)}")
    return matches[0]


def _runs_suffix(label):
    """The ' (N runs)' part load_results adds to labels, or '' if absent."""
    match = re.search(r" \(\d+ runs\)$", label)
    return match.group(0) if match else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append",
                        help="results file written by main.py; repeat to compare several files")
    parser.add_argument("--label", action="append",
                        help="legend name for the matching --file (give one per file, in the same order)")
    parser.add_argument("--method", default=None,
                        help='keep only the method whose name starts with this, e.g. "constant step size"')
    parser.add_argument("--title", default="Non-stationary 10-armed bandit")
    parser.add_argument("--window", type=int, default=1, help="moving-average window in steps (1 = raw)")
    parser.add_argument("--layout", default="overlay", choices=["overlay", "columns"])
    parser.add_argument("--save", default=None, help="save the figure to this path instead of showing it")
    args = parser.parse_args()

    files = args.file or ["results/hw1_bandit.npz"]
    if args.label and len(args.label) != len(files):
        parser.error(f"got {len(files)} --file but {len(args.label)} --label; give one label per file")

    combined = {}
    for i, path in enumerate(files):
        results, settings = load_results(path)
        print(f"{path}: {settings}")
        if args.method:
            key = _select_method(results, args.method, path)
            results = {key: results[key]}

        for key, data in results.items():
            if args.label:
                # your label, plus the method name only if this file still has several methods
                base = key[: len(key) - len(_runs_suffix(key))]
                name = args.label[i] if len(results) == 1 else f"{args.label[i]}: {base}"
                name += _runs_suffix(key)
            elif len(files) > 1:
                name = f"{key} [{os.path.splitext(os.path.basename(path))[0]}]"  # tell files apart
            else:
                name = key
            combined[name] = data

    plot_bandit_results(combined, window=args.window, layout=args.layout,
                        title=args.title, save_path=args.save)


if __name__ == "__main__":
    main()