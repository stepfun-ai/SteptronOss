# SteptronOss

[![Release](https://img.shields.io/github/v/release/Randomizez/SteptronOss)](https://img.shields.io/github/v/release/Randomizez/SteptronOss)
[![Build status](https://img.shields.io/github/actions/workflow/status/Randomizez/SteptronOss/main.yml?branch=main)](https://github.com/Randomizez/SteptronOss/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/Randomizez/SteptronOss)](https://img.shields.io/github/commit-activity/m/Randomizez/SteptronOss)
[![License](https://img.shields.io/github/license/Randomizez/SteptronOss)](https://img.shields.io/github/license/Randomizez/SteptronOss)

This is a template repository for Python projects that use uv for their dependency management.

## Runtime environment

The distributed helpers rely on a shared filesystem path for rendezvous when bringing up
per-experiment Redis servers. Set `STEPTRON_MEET_DIR` to a directory that is visible to
all participating nodes.
