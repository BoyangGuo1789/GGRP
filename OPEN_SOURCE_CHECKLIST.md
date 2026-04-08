# Open Source Release Checklist

## 1) Keep only source code and docs

- Remove tracked cache/packaging artifacts from git index:

```bash
git rm -r --cached Dassl.pytorch/dassl.egg-info
git ls-files | rg "__pycache__|\\.pyc$" | xargs git rm --cached
```

- Verify that generated result folders are not tracked:

```bash
git ls-files | rg "^(output|output_log|accuracy_csv|clip/accuracy_csv)/"
```

## 2) Check repository metadata

- Ensure root `README.md` matches the final paper title and method name.
- Add a root `LICENSE` file (choose one intentionally, e.g. MIT/Apache-2.0).
- Keep citation section as title-only until publication metadata is ready.

## 3) Script release policy

- Keep `scripts/ggrp/` private for now (do not add to git).
- Publish a separate open-source script set later.

## 4) Run a clean reproducibility check

- Create a fresh env and run one minimal training command.
- Run one evaluation command.
- Confirm no new tracked files appear after a run:

```bash
git status --short
```

## 5) Sanity checks before push

- Search for personal paths and secrets:

```bash
rg -n "/data/|/Users/|API_KEY|SECRET|TOKEN|PASSWORD"
```

- Check deleted/renamed files are intentional:

```bash
git status --short
```

- Review final diff:

```bash
git diff --stat
git diff
```
