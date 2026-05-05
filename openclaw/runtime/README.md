# Runtime Layout

This directory documents the expected layout of a local `~/.openclaw` runtime without committing private state.

The real local runtime usually contains items such as:

```text
~/.openclaw/
  openclaw.json
  completions/
  service-env/
  workspace/
  logs/
  tasks/
  flows/
  identity/
  devices/
  tmp/
  plugins/
```

This repository intentionally includes only the parts that are safe and useful to share:

- `workspace/` is versioned in this repo as `openclaw/workspace/`
- `config/openclaw.example.json` is the sanitized config template
- `runtime/completions/` contains shell completion references
- `runtime/service-env/` contains safe service environment examples

Not included:

- real tokens
- real `openclaw.json`
- device auth files
- logs
- SQLite state
- temporary files
- plugin runtime caches

