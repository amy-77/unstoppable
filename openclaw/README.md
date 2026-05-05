# OpenClaw Agent Framework

This directory is a shareable, safe OpenClaw agent project.

It is designed for teammates who want something more complete than a single skill folder:

- shared agent workspace files
- custom skills
- sanitized config template
- runtime layout references
- install / sync scripts

It is not a raw dump of one machine's private `~/.openclaw` directory.

## Project Structure

```text
openclaw/
  README.md
  .gitignore
  workspace/
    .gitignore
    AGENTS.md
    HEARTBEAT.md
    IDENTITY.md
    SOUL.md
    TOOLS.md
    USER.md
    skills/
      matchmaker/
        SKILL.md
  config/
    openclaw.example.json
  runtime/
    README.md
    completions/
    service-env/
  scripts/
    install.sh
    sync-workspace.sh
```

## What Is Shared

### `workspace/`

This is the actual shared agent workspace layer.

After installation, its contents should live in:

```bash
~/.openclaw/workspace
```

This includes:

- `AGENTS.md`
- `SOUL.md`
- `TOOLS.md`
- `IDENTITY.md`
- `USER.md`
- custom skills under `workspace/skills/`

### `config/`

This contains a sanitized runtime config template:

- `config/openclaw.example.json`

Teammates should copy it to:

```bash
~/.openclaw/openclaw.json
```

and then fill in their own provider keys and token values locally.

### `runtime/`

This documents the broader `~/.openclaw` layout and includes safe reference files such as:

- service environment examples
- shell completion placeholders

### `scripts/`

These scripts help teammates install or sync this project into a local OpenClaw setup.

## Current Reference Setup

The reference machine currently uses:

- config dir: `~/.openclaw`
- workspace dir: `~/.openclaw/workspace`
- gateway mode: `local`
- gateway port: `18789`
- tool profile: `coding`

Important distinction:

- `~/.openclaw/openclaw.json` is runtime config
- `~/.openclaw/workspace/` is shared workspace content

## Installation

### 1. Install OpenClaw locally

Make sure these directories exist:

```bash
ls ~/.openclaw
ls ~/.openclaw/workspace
```

### 2. Clone this repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_DIR>/openclaw
```

### 3. Install or sync the framework

Option A:

```bash
./scripts/install.sh
```

This will:

- sync `workspace/` into `~/.openclaw/workspace/`
- create `~/.openclaw/openclaw.json` from `config/openclaw.example.json` if it does not already exist

Option B:

```bash
./scripts/sync-workspace.sh
```

This only updates the workspace files.

## Skill Directory Rule

This is the most important rule in the project.

All shared custom skills must live under:

```bash
workspace/skills/
```

After syncing, OpenClaw must see them under:

```bash
~/.openclaw/workspace/skills/
```

Each skill should follow:

```text
workspace/skills/<skill-name>/SKILL.md
```

Current example:

```text
workspace/skills/matchmaker/SKILL.md
```

If teammates put skills elsewhere, the repo may look correct while the live OpenClaw workspace will not load them.

## Current Custom Skill

### `matchmaker`

Repo path:

```text
workspace/skills/matchmaker/SKILL.md
```

Live installed path:

```text
~/.openclaw/workspace/skills/matchmaker/SKILL.md
```

Purpose:

- matchmaking analysis
- dating market positioning
- partner selection strategy
- diagnosing mismatch between conditions and expectations

Trigger reminders are documented in:

- `workspace/TOOLS.md`

## What Is Intentionally Not Committed

This repository does not include:

- real `~/.openclaw/openclaw.json`
- real provider API keys
- real gateway token
- device auth files
- logs
- SQLite state files
- temporary runtime files
- plugin runtime caches

Those belong to each teammate's own local machine.

## Launch / Restart Notes

On the reference machine, OpenClaw is managed by `launchd`.

Useful commands:

```bash
launchctl print gui/$(id -u)/ai.openclaw.gateway
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.gateway.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.gateway.plist
```
