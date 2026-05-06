#!/usr/bin/env python3
"""
Refine: 根据诊断结果，用 patch 模式改进 SKILL.md、references 和 prompt 模板。

核心设计：不让 LLM 输出完整文件（会被 8192 token 截断），
而是输出结构化 patch 指令，程序逐个应用。

Usage:
    python3 scripts/refine.py --diagnosis scripts/iteration_history/diagnosis.json
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def get_client():
    try:
        import anthropic
    except ImportError:
        print("Error: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: need ANTHROPIC_API_KEY")
        sys.exit(1)

    kwargs = {"api_key": api_key}
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return anthropic.Anthropic(**kwargs)


def apply_patches(patches: list, base_dir: Path) -> list:
    """应用 patch 列表到文件系统"""
    results = []
    for i, patch in enumerate(patches):
        file_rel = patch.get("file", "")
        file_path = base_dir / file_rel
        action = patch.get("action", "")
        desc = patch.get("description", "")

        if not file_path.exists():
            results.append({"index": i, "file": file_rel, "action": action, "status": "file_not_found", "description": desc})
            continue

        content = file_path.read_text(encoding="utf-8")
        status = "applied"

        if action == "replace":
            old = patch.get("old", "")
            new = patch.get("new", "")
            if not old:
                status = "empty_old"
            elif old in content:
                content = content.replace(old, new, 1)
            else:
                # fuzzy: try stripping whitespace differences
                old_stripped = " ".join(old.split())
                content_stripped_check = " ".join(content.split())
                if old_stripped in content_stripped_check:
                    # find approximate location and replace
                    lines = content.split("\n")
                    old_lines = old.split("\n")
                    found = False
                    for j in range(len(lines) - len(old_lines) + 1):
                        chunk = "\n".join(lines[j:j + len(old_lines)])
                        if " ".join(chunk.split()) == old_stripped:
                            content = content.replace(chunk, new, 1)
                            found = True
                            break
                    status = "applied_fuzzy" if found else "anchor_not_found"
                else:
                    status = "anchor_not_found"

        elif action == "insert_after":
            anchor = patch.get("anchor", "")
            insert_content = patch.get("content", "")
            if anchor and anchor in content:
                idx = content.index(anchor) + len(anchor)
                # find end of the anchor line
                newline_idx = content.find("\n", idx)
                if newline_idx == -1:
                    content = content + "\n" + insert_content
                else:
                    content = content[:newline_idx] + "\n" + insert_content + content[newline_idx:]
            else:
                status = "anchor_not_found"

        elif action == "insert_before":
            anchor = patch.get("anchor", "")
            insert_content = patch.get("content", "")
            if anchor and anchor in content:
                idx = content.index(anchor)
                content = content[:idx] + insert_content + "\n" + content[idx:]
            else:
                status = "anchor_not_found"

        elif action == "append":
            insert_content = patch.get("content", "")
            content = content.rstrip() + "\n\n" + insert_content + "\n"

        elif action == "prepend":
            insert_content = patch.get("content", "")
            content = insert_content + "\n\n" + content

        else:
            status = f"unknown_action:{action}"

        if status in ("applied", "applied_fuzzy"):
            file_path.write_text(content, encoding="utf-8")

        results.append({"index": i, "file": file_rel, "action": action, "status": status, "description": desc})

    return results


def _fix_json_string(text: str) -> str:
    """Fix common LLM JSON issues: unescaped newlines/tabs inside string values"""
    result = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and (i == 0 or text[i-1] != '\\'):
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == '\n':
            result.append('\\n')
        elif in_string and ch == '\t':
            result.append('\\t')
        else:
            result.append(ch)
        i += 1
    return ''.join(result)


def generate_patches_for_target(client, model: str, target_file: str, current_content: str,
                                 improvements: list, diagnosis: dict) -> list:
    """让 LLM 生成针对单个文件的 patch 指令"""
    prompt = f"""你是 Skill 优化专家。根据改进建议，生成对 {target_file} 的修改补丁。

## 改进要求
{json.dumps(improvements, ensure_ascii=False, indent=2)}

## 诊断上下文
- 最弱维度: {diagnosis.get('weakest_dimension', 'unknown')}
- 根因: {json.dumps(diagnosis.get('root_causes', []), ensure_ascii=False)}
- 缺失原型: {json.dumps(diagnosis.get('missing_archetypes', []), ensure_ascii=False)}

## 当前文件内容
```
{current_content}
```

## 输出格式

输出一个 JSON 数组，每个元素是一个 patch 操作。用 ```json 包裹。

支持的 action:
- "replace": 精确替换。需要 "old"（原文片段，足够长以唯一定位）和 "new"（替换后内容）
- "insert_after": 在 anchor 之后插入。需要 "anchor"（原文中的一行或片段）和 "content"（要插入的内容）
- "insert_before": 在 anchor 之前插入。需要 "anchor" 和 "content"
- "append": 追加到文件末尾。需要 "content"

```json
[
  {{
    "action": "replace",
    "old": "要替换的原文片段（必须在文件中精确存在，包括换行和缩进）",
    "new": "替换后的新内容",
    "description": "简述这个 patch 做了什么"
  }},
  {{
    "action": "insert_after",
    "anchor": "定位用的原文片段（必须精确匹配）",
    "content": "要插入的新内容",
    "description": "简述"
  }}
]
```

关键规则：
1. old/anchor 必须是文件中**精确存在**的文本，包括空格和换行
2. 每个 patch 独立，不依赖其他 patch 的执行结果
3. 保持文件整体结构和风格一致
4. 优先用 insert_after/insert_before 添加新内容，用 replace 修改现有内容
5. 不要输出超大的 patch（单个 patch 的 new/content 不超过 500 字）
6. JSON 字符串中的换行必须用 \\n 表示，不能有裸换行
7. 如果改进内容过多，优先保留最高优先级的 patch，宁可少输出也不要截断
8. **严格要求**：JSON 输出中所有标点符号必须使用英文标点(双引号, 逗号, 冒号, 方括号, 花括号)。content/new 字段的文本内容中也只使用英文标点,不要使用中文引号、中文逗号、中文冒号等"""

    msg = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = msg.content[0].text

    if msg.stop_reason == "max_tokens":
        print(f"  [Patch] Warning: response truncated (max_tokens) for {target_file}")
        print(f"  [Patch] Raw response (last 200 chars): {text[-200:]}")
        return []

    # extract JSON array — try multiple strategies
    patches = None

    # Strategy 1: strip code fences, find the JSON array
    clean = text
    # Remove opening fence
    clean = re.sub(r'```(?:json)?\s*\n?', '', clean)
    # Remove closing fence
    clean = re.sub(r'\n?```', '', clean)

    # Try parsing the cleaned text directly
    clean = clean.strip()
    if clean.startswith('['):
        try:
            patches = json.loads(clean)
        except json.JSONDecodeError as e:
            print(f"  [Patch] Strategy 1 parse error: {e}")
            # Fix common LLM JSON issues: control chars inside strings
            fixed = _fix_json_string(clean)
            try:
                patches = json.loads(fixed)
                print(f"  [Patch] Strategy 1b (fixed) succeeded")
            except json.JSONDecodeError:
                # Try to find the last valid ] that closes the array
                for i in range(len(fixed) - 1, 0, -1):
                    if fixed[i] == ']':
                        try:
                            patches = json.loads(fixed[:i+1])
                            break
                        except json.JSONDecodeError:
                            continue

    # Strategy 2: regex fallback for embedded JSON
    if not patches:
        m = re.search(r'(\[\s*\{[\s\S]*\}\s*\])', text)
        if m:
            try:
                patches = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Strategy 3: single object
    if not patches:
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "action" in obj:
                    patches = [obj]
            except json.JSONDecodeError:
                pass

    if patches and isinstance(patches, list):
        for p in patches:
            p["file"] = target_file
        return patches

    print(f"  [Patch] Warning: could not parse patches for {target_file}")
    print(f"  [Patch] Raw response (first 500 chars): {text[:500]}")
    print(f"  [Patch] Raw response (last 200 chars): {text[-200:]}")
    return []


def refine_skill(client, model: str, skill_path: Path, improvements: list, diagnosis: dict) -> dict:
    """用 patch 模式改进 SKILL.md"""
    skill_improvements = [imp for imp in improvements
                          if imp.get("target") == "skill_md" and imp.get("priority", 3) <= 2]

    if not skill_improvements:
        print("  [Skill] No high-priority improvements needed")
        return {"skill_md": "no_change"}

    skill_improvements = sorted(skill_improvements, key=lambda x: x.get("priority", 3))[:5]
    skill_text = skill_path.read_text(encoding="utf-8")
    rel_path = str(skill_path.relative_to(BASE_DIR))

    print(f"  [Skill] Generating patches for {len(skill_improvements)} improvements...")
    patches = generate_patches_for_target(client, model, rel_path, skill_text, skill_improvements, diagnosis)

    if not patches:
        return {"skill_md": "no_patches_generated"}

    print(f"  [Skill] Applying {len(patches)} patches...")
    results = apply_patches(patches, BASE_DIR)

    applied = sum(1 for r in results if r["status"] in ("applied", "applied_fuzzy"))
    failed = sum(1 for r in results if r["status"] not in ("applied", "applied_fuzzy"))
    print(f"  [Skill] Applied: {applied}, Failed: {failed}")

    for r in results:
        status_icon = "✓" if r["status"] in ("applied", "applied_fuzzy") else "✗"
        print(f"    {status_icon} {r['description']} [{r['status']}]")

    if applied > 0:
        print(f"  [Skill] Updated: {skill_path}")

    return {"skill_md": "updated", "patches_applied": applied, "patches_failed": failed, "details": results}


def refine_references(client, model: str, skill_dir: Path, improvements: list, diagnosis: dict) -> dict:
    """用 patch 模式改进 references 目录下的所有文件"""
    ref_improvements = [imp for imp in improvements
                        if imp.get("target") == "references" and imp.get("priority", 3) <= 2]

    if not ref_improvements:
        print("  [Refs] No high-priority improvements needed")
        return {"references": "no_change"}

    ref_dir = skill_dir / "references"
    if not ref_dir.exists():
        print("  [Refs] references/ directory not found")
        return {"references": "not_found"}

    # group improvements by target file (infer from section field)
    file_mapping = {
        "pattern_library": "pattern_library.md",
        "case_archive": "case_archive.md",
        "case_digest": "case_digest.md",
        "statistics_snapshot": "statistics_snapshot.md",
    }

    all_results = {}
    # default: apply all ref improvements to pattern_library unless section specifies otherwise
    target_files = set()
    for imp in ref_improvements:
        section = imp.get("section", "pattern_library")
        for key, fname in file_mapping.items():
            if key in section:
                target_files.add(fname)
                break
        else:
            target_files.add("pattern_library.md")

    for fname in target_files:
        fpath = ref_dir / fname
        if not fpath.exists():
            all_results[fname] = "file_not_found"
            continue

        content = fpath.read_text(encoding="utf-8")
        rel_path = str(fpath.relative_to(BASE_DIR))

        # filter improvements relevant to this file
        relevant_imps = []
        for imp in ref_improvements:
            section = imp.get("section", "")
            mapped = False
            for key, fn in file_mapping.items():
                if key in section and fn == fname:
                    relevant_imps.append(imp)
                    mapped = True
                    break
            if not mapped and fname == "pattern_library.md":
                relevant_imps.append(imp)

        if not relevant_imps:
            continue

        print(f"  [Refs] Generating patches for {fname} ({len(relevant_imps)} improvements)...")
        patches = generate_patches_for_target(client, model, rel_path, content, relevant_imps, diagnosis)

        if patches:
            results = apply_patches(patches, BASE_DIR)
            applied = sum(1 for r in results if r["status"] in ("applied", "applied_fuzzy"))
            print(f"  [Refs] {fname}: {applied}/{len(patches)} patches applied")
            all_results[fname] = {"applied": applied, "total": len(patches)}
        else:
            all_results[fname] = "no_patches_generated"

    return {"references": "updated", "details": all_results}


def refine_prompts(client, model: str, improvements: list, diagnosis: dict) -> dict:
    """用 patch 模式改进 i18n.py 中的 prompt 模板"""
    prompt_improvements = [imp for imp in improvements
                           if imp.get("target") == "prompts" and imp.get("priority", 3) <= 2]

    if not prompt_improvements:
        print("  [Prompts] No prompt improvements needed")
        return {"prompts": "no_change"}

    # Prompts live in i18n.py
    i18n_path = BASE_DIR / "scripts" / "i18n.py"
    if not i18n_path.exists():
        print("  [Prompts] i18n.py not found")
        return {"prompts": "not_found"}

    content = i18n_path.read_text(encoding="utf-8")
    rel_path = str(i18n_path.relative_to(BASE_DIR))

    print(f"  [Prompts] Generating patches for {len(prompt_improvements)} improvements...")
    patches = generate_patches_for_target(client, model, rel_path, content, prompt_improvements, diagnosis)

    if not patches:
        print("  [Prompts] No patches generated")
        return {"prompts": "no_patches_generated"}

    results = apply_patches(patches, BASE_DIR)
    applied = sum(1 for r in results if r["status"] in ("applied", "applied_fuzzy"))
    failed = sum(1 for r in results if r["status"] not in ("applied", "applied_fuzzy"))
    print(f"  [Prompts] Applied: {applied}, Failed: {failed}")

    for r in results:
        status_icon = "✓" if r["status"] in ("applied", "applied_fuzzy") else "✗"
        print(f"    {status_icon} {r['description']} [{r['status']}]")

    return {"prompts": "updated", "patches_applied": applied, "patches_failed": failed, "details": results}


def backup_files(skill_path: Path, round_dir: Path):
    """备份所有可能被修改的文件"""
    backup_dir = round_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # backup SKILL.md
    shutil.copy2(skill_path, round_dir / "SKILL.md.bak")

    # backup references
    ref_dir = skill_path.parent / "references"
    if ref_dir.exists():
        ref_backup = backup_dir / "references"
        ref_backup.mkdir(exist_ok=True)
        for f in ref_dir.glob("*.md"):
            shutil.copy2(f, ref_backup / f.name)

    # backup distill_pipeline.py
    pipeline_path = BASE_DIR / "scripts" / "distill_pipeline.py"
    if pipeline_path.exists():
        shutil.copy2(pipeline_path, backup_dir / "distill_pipeline.py.bak")

    # backup i18n.py (prompt templates)
    i18n_path = BASE_DIR / "scripts" / "i18n.py"
    if i18n_path.exists():
        shutil.copy2(i18n_path, backup_dir / "i18n.py.bak")

    print(f"[Refine] Backups saved to: {backup_dir}")


def run_refinement(diagnosis_path: Path, skill_path: Path, round_dir: Path) -> dict:
    client = get_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    print(f"[Refine] Model: {model}")

    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    improvements = diagnosis.get("improvements", [])

    if not improvements:
        print("[Refine] No improvements to apply")
        return {"status": "no_improvements"}

    # backup all files
    backup_files(skill_path, round_dir)

    results = {}
    results["skill"] = refine_skill(client, model, skill_path, improvements, diagnosis)
    results["references"] = refine_references(client, model, skill_path.parent, improvements, diagnosis)
    results["prompts"] = refine_prompts(client, model, improvements, diagnosis)

    return results


def main():
    parser = argparse.ArgumentParser(description="Refine skill based on diagnosis (patch mode)")
    parser.add_argument("--diagnosis", required=True, help="Path to diagnosis.json")
    parser.add_argument("--skill-path", default="zh/SKILL.md", help="Path to SKILL.md")
    parser.add_argument("--round-dir", default=None, help="Directory for this iteration round")
    args = parser.parse_args()

    diagnosis_path = Path(args.diagnosis)
    if not diagnosis_path.exists():
        print(f"Error: {diagnosis_path} not found")
        sys.exit(1)

    skill_path = BASE_DIR / args.skill_path
    round_dir = Path(args.round_dir) if args.round_dir else diagnosis_path.parent
    round_dir.mkdir(parents=True, exist_ok=True)

    results = run_refinement(diagnosis_path, skill_path, round_dir)

    out_path = round_dir / "refine_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Refine] Results: {out_path}")


if __name__ == "__main__":
    main()
