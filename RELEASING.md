# Releasing `thermologic` to PyPI

The package is build-ready and validated (`python -m build` + `twine check` pass;
it installs in a clean venv and imports). These are the steps to publish.

## One-time setup
1. Create a **PyPI account** and an **API token** (Account → API tokens).
2. Check the name is free: <https://pypi.org/project/thermologic/>. If taken,
   change `name` in `pyproject.toml` (e.g. `thermologic-ai`) — the import package
   stays `thermologic`.
3. Use a **recent** toolchain (system Debian tooling is too old):
   ```bash
   python -m venv .release && source .release/bin/activate
   pip install -U pip build twine
   ```

## Each release
1. Bump the version in **two** places (keep them in sync):
   - `pyproject.toml` → `version`
   - `thermologic/__init__.py` → `__version__`
2. Build and check:
   ```bash
   rm -rf dist build *.egg-info
   python -m build          # -> dist/thermologic-X.Y.Z-py3-none-any.whl + .tar.gz
   twine check dist/*       # must say PASSED for both
   ```
3. (Recommended) Dry-run on **TestPyPI** first:
   ```bash
   twine upload --repository testpypi dist/*
   pip install -i https://test.pypi.org/simple/ thermologic
   ```
4. Publish:
   ```bash
   twine upload dist/*      # paste your API token when prompted (user: __token__)
   ```
5. Verify from a clean environment:
   ```bash
   pip install thermologic
   python -c "from thermologic import LogicEnergy; print('ok')"
   ```
6. Tag the release in git:
   ```bash
   git tag -a vX.Y.Z -m "thermologic X.Y.Z" && git push --tags
   ```

## Notes
- The wheel ships only the `thermologic/` package (typed, with `py.typed`); the
  research scripts (`experiments*.py`, `benchmarks/`, `train.py`) are intentionally
  **not** part of the distributable.
- `torch` is the only hard dependency; `numpy`/`matplotlib` are `[experiments]`
  extras and `build`/`twine`/`pytest` are `[dev]` extras.
- For a polished PyPI project page, switch the README's hero image to an absolute
  URL (`https://raw.githubusercontent.com/joperjoker/Project-ThermoLogic/main/assets/playground_preview.png`)
  once the branch is merged to `main` — relative image paths do not render on PyPI.
