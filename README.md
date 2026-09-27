# mini-dataloader

A learning project built to understand core Python (OOP, decorators, generators, context managers, typing, dataclasses, abc) by building the infrastructure around a PyTorch training pipeline from scratch, then connecting it to a real model trained on FashionMNIST.

## Project goal

This project exists to internalize how PyTorch works "under the hood". Concretely, it's meant to give hands-on practice with:

- `abc` — abstract base classes (`Dataset`)
- generators - custom `DataLoader` batching via `__iter__`/`yield`
- `dataclasses` + `typing`
- decorators — timing/logging around training steps
- context managers
- a small test framework (`pytest`) covering the above
- pre-commit hooks for formatting, linting, typing, and running tests before every commit

## Features

- loads FashionMNIST data from the `db/` directory,
- custom `DataLoader` class (generator-based, `abc`-backed `Dataset`) with batching support,
- simple `Trainer` for training and validation,
- configurable `TrainingConfig` (`dataclass`, typed),
- ready-to-use CNN example (real `torch.nn.Module`, `autograd`, `optim`),
- factory functions for `criterion` and `optimizer`,
- logging and timing decorators,
- pre-commit hooks (formatting, linting, type-checking, tests) enforcing code quality on every commit.
