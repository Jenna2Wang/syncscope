# Contributing

Thanks for taking the time to contribute! This project is small and the bar is
simple: keep it correct, typed, and tested.

## Development setup

```bash
git clone https://github.com/stella-sage553/syncscope
cd syncscope
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

```bash
ruff check .
ruff format --check .
mypy syncscope
pytest --cov=syncscope
```

The same checks run in CI across Python 3.10–3.13, so green locally usually means
green on the PR.

## Guidelines

- **One concern per PR.** Small, focused changes are easier to review and revert.
- **Add a test** for any behaviour change. The suite runs on synthetic signals,
  so there is no need to add media files.
- **Keep the core NumPy-only.** Anything that needs librosa or OpenCV belongs
  behind the `audio` / `video` extras and a lazy import.
- **Type everything.** Public functions are fully annotated; `mypy` is part of CI.
- Update `CHANGELOG.md` under `[Unreleased]` when you change behaviour.

## Reporting bugs

Open an issue with a minimal snippet that reproduces the problem (the synthetic
generators in `syncscope.synthetic` are handy for this).
