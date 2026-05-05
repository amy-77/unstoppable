# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

### 婚恋市场洞察 Skill
- 名称：matchmaking-insight-zh
- 路径：~/.openclaw/workspace/skills/matchmaker/SKILL.md
- 触发关键词：相亲分析、婚恋市场、找对象、匹配策略、市场价值、帮我分析一下条件、我该找什么样的、为什么找不到对象
- **一旦触发，必须先读 SKILL.md，然后按框架分析，开头写 [MATCHMAKING_SKILL_ACTIVE]，并打印「激活skill：~/.openclaw/workspace/skills/matchmaker/SKILL.md」**

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.

## Related

- [Agent workspace](/concepts/agent-workspace)
