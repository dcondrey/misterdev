# Evaluation

Harnesses that measure misterdev against public benchmarks — turning "is this
change an improvement?" into a number.

## SWE-bench

Run misterdev on real GitHub-issue tasks and grade each patch against the task's
own hidden tests. A task is **resolved** when, after the model's patch and the
task's `test_patch` are applied, every `FAIL_TO_PASS` test passes and every
`PASS_TO_PASS` test still passes.

### How it works

misterdev edits files on the host, but the build/test gates run **inside the
task's official SWE-bench Docker image** (bind-mounted at `/testbed`, where that
image's editable install resolves) via misterdev's container execution env. So a
task runs in its exact dependency environment without installing misterdev in the
container.

- `instance.py` — the task record (tolerant of the dataset's field shapes).
- `grader.py` — the ground truth: applies `test_patch`, runs the tests, decides
  `resolved`. Dependency-free and unit-tested.
- `runner.py` — set up the repo at `base_commit`, drive `misterdev build` with
  the issue as the goal, extract and grade the patch. Repo setup is injectable.
- `docker_runner.py` — derive the instance image, write the container-env
  `project.yaml`, and grade inside the image.
- `harness.py` / `__main__.py` — run a suite and report the resolved rate.

### Requirements

- Docker (with `linux/amd64` emulation available on Apple Silicon)
- `pip install swebench datasets`
- An API key for the provider your run uses (`OPENROUTER_API_KEY` /
  `ANTHROPIC_API_KEY`)

### Run

```bash
# 1. Export a slice of the dataset to JSONL (one record per line), then:
python -m evaluation.swebench \
  --dataset swe_lite.jsonl \
  --workdir /tmp/swe \
  --limit 5 \
  --build-args "--budget 5 --allow-dirty --no-suggest"
```

The suite prints `N/M resolved (rate)` plus a per-instance pass/fail line.

### Cost & time

Each instance is one misterdev build (~$2–5 with a strong model; less with
caching and free-model routing). Wall-clock is dominated by Docker image
build/run under emulation on Apple Silicon — run on a native x86_64 machine for a
full slice. Start with `--limit` to measure a real per-instance cost before
committing to a large run.
