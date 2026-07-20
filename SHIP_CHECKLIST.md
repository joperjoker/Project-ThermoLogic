# Ship checklist

Everything in this repo is build-ready. The three publishing steps below each
need **your** account/credentials, so they can't be done from the sandbox — here
is exactly what to run/click.

## 1. PyPI (the `thermologic` library)
The package is validated (`python -m build` + `twine check` pass; imports in a
clean venv). Full steps are in [`RELEASING.md`](RELEASING.md). Short version:
```bash
python -m venv .release && source .release/bin/activate
pip install -U pip build twine
rm -rf dist build *.egg-info
python -m build           # -> dist/thermologic-0.1.0-{whl,tar.gz}
twine check dist/*        # must say PASSED
twine upload dist/*       # user: __token__, paste your PyPI API token
```
- Only `thermologic/` ships (typed, `py.typed`); research scripts are excluded on
  purpose.
- If the name `thermologic` is taken, change `name` in `pyproject.toml` (e.g.
  `thermologic-ai`); the import package stays `thermologic`.

## 2. GitHub Pages (the interactive playground)
`index.html` is the Pages entry point at the repo root.
1. Merge `claude/thermologic-neuro-symbolic-7vhmbh` into your default branch (or
   point Pages at this branch).
2. GitHub → repo **Settings → Pages** → Source = "Deploy from a branch", branch =
   `main` (or the chosen branch), folder = `/ (root)` → **Save**.
3. Live in ~1 min at `https://joperjoker.github.io/Project-ThermoLogic/`
   (already wired as the `Demo` URL in `pyproject.toml`).

## 3. LinkedIn post
- **PDF (academic report):** `ThermoLogic_TechnicalReport.pdf` — regenerated, now
  includes the Semantic Loss novelty check, measured scaling, and the honest
  external Latin-square loss.
- **Slides (carousel):** `ThermoLogic_LinkedIn_Slides.pdf` (1080×1080).
- **Draft copy:** `LINKEDIN_POST.md`.
- Author line and profile (`www.linkedin.com/in/eugene-teo`) are already set.

> Note: the carousel slides predate the four new findings (§5.11, §6.3–§6.5). If
> you want them reflected on the slides, say the word and I'll refresh the deck;
> the PDF report and the paper are already fully up to date.

## What each headline claim rests on (so you can defend it)
| Claim | Evidence | Honest caveat |
|---|---|---|
| Label-free consistency guarantee | §5.3, §6.2 | detects inconsistency, not general correctness |
| Repair scales where projection can't | §6.4 (measured to 9×9 Sudoku) | a real CP/SAT solver still scales |
| Semi-supervised win is real & novel | §5.10, §5.11 (beats Semantic Loss, weight-robust) | one task; no general-superiority claim |
| We report where it loses | §6.3 (external Latin squares) | a solver dominates small clean problems |
