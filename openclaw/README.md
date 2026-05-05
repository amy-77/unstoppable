# OpenClaw Workspace

This repository contains our shared OpenClaw workspace setup, prompt files, and custom skills.

It is intended to let teammates:

- clone the project from GitHub
- place the files into the correct local OpenClaw workspace
- use the same agent behavior and custom skills
- continue adding or updating shared skills in one place

## What Is In This Repo

This repo currently focuses on the **workspace layer**, not personal secrets.

Included:

- `AGENTS.md`: workspace operating rules
- `IDENTITY.md`: agent identity/personality
- `USER.md`: user-specific context
- `TOOLS.md`: local notes and skill trigger reminders
- `HEARTBEAT.md`: heartbeat checklist
- `SOUL.md`: higher-level style / behavior notes
- `skills/`: custom skills shared by the team
- `memory/`: local memory notes if we decide to keep them in repo

Not recommended to commit:

- API keys
- personal tokens
- private device auth files
- `~/.openclaw/openclaw.json` with real secrets inside

## Current Setup

Our current active OpenClaw instance is running locally on macOS with:

- OpenClaw config dir: `~/.openclaw`
- Active workspace: `~/.openclaw/workspace`
- Gateway mode: `local`
- Gateway port: `18789`
- Managed by: `launchd`
- Current tool profile: `coding`

Important distinction:

- `~/.openclaw/openclaw.json` is the **runtime config**
- `~/.openclaw/workspace/` is the **shared workspace content**

This repo is mainly for the second part: the shared workspace content.

## Recommended Team Setup

### 1. Install OpenClaw

Each teammate should install OpenClaw locally first.

Then confirm their default workspace exists:

```bash
ls ~/.openclaw/workspace
```

### 2. Clone This Repo

Clone this repository anywhere first:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_REPO_DIR>
```

### 3. Copy Or Sync Into The Real OpenClaw Workspace

OpenClaw reads the active workspace from:

```bash
~/.openclaw/workspace
```

So teammates should copy this repo's contents into that directory, or make this repo itself become that directory.

Simplest approach:

```bash
cp -R . ~/.openclaw/workspace/
```

If they want this repo to be the live workspace directly, they can also clone it to:

```bash
~/.openclaw/workspace
```

## Skill Directory Rule

This is the most important convention in this project.

**All shared custom skills must be placed under:**

```bash
~/.openclaw/workspace/skills/
```

Each skill should have its own folder, for example:

```text
~/.openclaw/workspace/skills/matchmaker/SKILL.md
```

That means:

- skill root directory: `skills/<skill-name>/`
- required entry file: `SKILL.md`

For our current custom skill:

- folder: `skills/matchmaker/`
- file: `skills/matchmaker/SKILL.md`
- skill name inside file: `matchmaking-insight-zh`

If a teammate puts the skill somewhere else, OpenClaw may not load it when the workspace is active.

## Current Custom Skill

### `matchmaker`

Path:

```text
skills/matchmaker/SKILL.md
```

Purpose:

- matchmaking analysis
- dating market positioning
- partner selection strategy
- diagnosing mismatch between personal conditions and expectations

The current trigger hints are also documented in `TOOLS.md`.

## Suggested Workflow For Adding A New Skill

1. Create a new folder under `skills/`
2. Add a `SKILL.md`
3. Document trigger reminders in `TOOLS.md` if needed
4. Test it locally in the active OpenClaw workspace
5. Commit and push to GitHub
6. Teammates pull and sync to their own `~/.openclaw/workspace`

Example:

```text
skills/
  my-new-skill/
    SKILL.md
```

## Files Teammates May Need To Adjust Locally

These files may contain personal or machine-specific preferences and should be reviewed after pulling:

- `USER.md`
- `TOOLS.md`
- `IDENTITY.md`
- `memory/`

These files should be handled carefully and not forced if they contain personal context.

## Runtime Config Notes

Our active runtime config is outside this repo's main purpose:

```text
~/.openclaw/openclaw.json
```

That file controls things like:

- model providers
- gateway auth
- port binding
- tool profile
- plugin enablement

Do **not** commit real secrets from that file to GitHub.

If needed, we can later add a sanitized example such as:

- `openclaw.example.json`

instead of sharing real tokens.

In this repo, teammates should use:

- `openclaw.example.json`

as the reference template, then create or update their own local:

- `~/.openclaw/openclaw.json`

## Launch / Restart

On the current machine, OpenClaw is managed by `launchd`.

Useful commands:

```bash
launchctl print gui/$(id -u)/ai.openclaw.gateway
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.gateway.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.openclaw.gateway.plist
```

## Collaboration Notes

- Keep shared skills in `skills/`
- Do not commit secrets
- Prefer documenting team-wide behavior in `AGENTS.md`
- Prefer documenting skill-specific usage in each skill's `SKILL.md`
- Prefer documenting local trigger reminders in `TOOLS.md`

## Recommended Next Improvement

Before publishing this repo, consider adding:

1. a short changelog for shared skill updates
2. a simple install script to sync the repo into `~/.openclaw/workspace`
3. a team convention for which files are shared vs personal
