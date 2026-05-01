#!/usr/bin/env python3
"""
Matchmaking Data Distillation Pipeline (multi-language)
Distills a "Matchmaking Market Insight" Skill from real matchmaking cases.

Usage:
    python3 scripts/distill_pipeline.py --lang zh              # Chinese (default)
    python3 scripts/distill_pipeline.py --lang en              # English
    python3 scripts/distill_pipeline.py --lang all             # Both languages
    python3 scripts/distill_pipeline.py --lang zh --skip-llm   # Stats only
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR.parent / "datasets"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from i18n import I18N


# ============================================================
# Step 1: Data Cleaning
# ============================================================

def fix_json(raw: str) -> list:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw = re.sub(r',(\s*[}\]])', r'\1', raw)
        raw = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', raw)
        return json.loads(raw)


def normalize_tier(tier, tier_map):
    return tier_map.get(tier, tier or tier_map.get(None, "Unknown"))


def normalize_income(income_tier, labels):
    if not income_tier:
        return labels["unknown"]
    base = re.match(r'A(\d+)', str(income_tier))
    if not base:
        return labels["unknown"]
    level = int(base.group(1))
    if level <= 4:
        return labels["low"]
    elif level <= 5:
        return labels["mid"]
    elif level <= 6:
        return labels["mid_high"]
    elif level <= 7:
        return labels["high"]
    else:
        return labels["top"]


def normalize_education(edu, cfg):
    if not edu:
        return cfg["edu_labels"]["unknown"]
    edu = str(edu)
    for level in ("phd", "master", "bachelor", "below"):
        if any(k in edu for k in cfg["edu_keywords"][level]):
            return cfg["edu_labels"][level]
    return cfg["edu_labels"]["bachelor"]


def clean_data(cases: list, cfg: dict) -> list:
    for c in cases:
        loc = c["subject_profile"]["location"]
        loc["tier_normalized"] = normalize_tier(loc.get("tier"), cfg["tier_map"])
        fin = c["subject_profile"]["financials"]
        fin["income_normalized"] = normalize_income(fin.get("income_tier"), cfg["income_labels"])
        c["subject_profile"]["education_normalized"] = normalize_education(
            c["subject_profile"].get("education"), cfg
        )
    return cases


def step1_clean(source_path: Path, data_dir: Path, cfg: dict) -> list:
    print("[Step 1] Data cleaning...")
    raw = source_path.read_text(encoding="utf-8")
    cases = fix_json(raw)
    cases = [c for c in cases if "matchmaking_intelligence" in c]
    cases = clean_data(cases, cfg)
    out_path = data_dir / "cleaned_data.json"
    out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Done: {len(cases)} cases → {out_path}")
    return cases


# ============================================================
# Step 2: Statistical Analysis
# ============================================================

def step2_analyze(cases: list, data_dir: Path, cfg: dict) -> dict:
    print("[Step 2] Statistical analysis...")
    report = {}

    report["total"] = len(cases)
    report["gender"] = dict(Counter(c["subject_profile"]["gender"] for c in cases))
    report["age_distribution"] = dict(Counter(c["subject_profile"]["age"] for c in cases))
    ages = [c["subject_profile"]["age"] for c in cases if c["subject_profile"]["age"]]
    report["age_stats"] = {"min": min(ages), "max": max(ages), "avg": round(sum(ages)/len(ages), 1)}
    report["city_tier"] = dict(Counter(
        c["subject_profile"]["location"]["tier_normalized"] for c in cases
    ).most_common())
    report["income"] = dict(Counter(
        c["subject_profile"]["financials"]["income_normalized"] for c in cases
    ).most_common())
    report["education"] = dict(Counter(
        c["subject_profile"]["education_normalized"] for c in cases
    ).most_common())

    all_tags = []
    tag_by_gender = defaultdict(list)
    for c in cases:
        tags = c["psychology_and_traits"]["personality_tags"]
        all_tags.extend(tags)
        tag_by_gender[c["subject_profile"]["gender"]].extend(tags)

    report["top_personality_tags"] = dict(Counter(all_tags).most_common(30))
    report["male_top_tags"] = dict(Counter(tag_by_gender["male"]).most_common(15))
    report["female_top_tags"] = dict(Counter(tag_by_gender["female"]).most_common(15))

    cooccurrence = Counter()
    for c in cases:
        tags = sorted(set(c["psychology_and_traits"]["personality_tags"]))
        for i in range(len(tags)):
            for j in range(i+1, len(tags)):
                cooccurrence[(tags[i], tags[j])] += 1
    report["tag_cooccurrence_top20"] = [
        {"pair": list(pair), "count": count}
        for pair, count in cooccurrence.most_common(20)
    ]

    kw_regex = cfg.get("conflict_keyword_regex", r'\b\w{3,}\b')
    conflict_keywords = Counter()
    for c in cases:
        conflict = c["matchmaking_intelligence"]["core_conflict"]
        words = re.findall(kw_regex, conflict)
        conflict_keywords.update(words)
    report["conflict_keywords_top30"] = dict(conflict_keywords.most_common(30))

    strategy_keywords = Counter()
    for c in cases:
        strategy = c["matchmaking_intelligence"]["expert_strategy"]
        words = re.findall(kw_regex, strategy)
        strategy_keywords.update(words)
    report["strategy_keywords_top30"] = dict(strategy_keywords.most_common(30))

    conflict_patterns = defaultdict(list)
    pattern_rules = cfg["pattern_rules"]
    other_label = cfg["other_label"]
    for c in cases:
        conflict = c["matchmaking_intelligence"]["core_conflict"]
        matched = False
        for pattern_name, keywords in pattern_rules.items():
            if any(kw in conflict for kw in keywords):
                conflict_patterns[pattern_name].append(c["metadata"]["case_id"])
                matched = True
                break
        if not matched:
            conflict_patterns[other_label].append(c["metadata"]["case_id"])
    report["conflict_patterns"] = {k: {"count": len(v), "cases": v[:5]} for k, v in
                                    sorted(conflict_patterns.items(), key=lambda x: -len(x[1]))}

    male_cases = [c for c in cases if c["subject_profile"]["gender"] == "male"]
    female_cases = [c for c in cases if c["subject_profile"]["gender"] == "female"]
    report["gender_analysis"] = {
        "male": {
            "count": len(male_cases),
            "avg_age": round(sum(c["subject_profile"]["age"] for c in male_cases if c["subject_profile"]["age"]) / max(1, sum(1 for c in male_cases if c["subject_profile"]["age"])), 1),
            "top_income": dict(Counter(c["subject_profile"]["financials"]["income_normalized"] for c in male_cases).most_common(5)),
        },
        "female": {
            "count": len(female_cases),
            "avg_age": round(sum(c["subject_profile"]["age"] for c in female_cases if c["subject_profile"]["age"]) / max(1, sum(1 for c in female_cases if c["subject_profile"]["age"])), 1),
            "top_income": dict(Counter(c["subject_profile"]["financials"]["income_normalized"] for c in female_cases).most_common(5)),
        }
    }

    representative_cases = []
    for c in cases:
        representative_cases.append({
            "case_id": c["metadata"]["case_id"],
            "gender": c["subject_profile"]["gender"],
            "age": c["subject_profile"]["age"],
            "income": c["subject_profile"]["financials"]["income_normalized"],
            "core_conflict": c["matchmaking_intelligence"]["core_conflict"],
            "strategy": c["matchmaking_intelligence"]["expert_strategy"][:100],
        })
    report["all_cases_summary"] = representative_cases

    out_path = data_dir / "statistics_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Done → {out_path}")
    return report


# ============================================================
# Step 3: LLM Extraction
# ============================================================

def step3_extract(cases: list, report: dict, api_key: str, data_dir: Path, cfg: dict) -> dict:
    print("[Step 3] LLM extraction...")
    try:
        import anthropic
    except ImportError:
        print("  Error: pip install anthropic")
        sys.exit(1)

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    print(f"  Model: {model}")

    stats_str = json.dumps({
        k: v for k, v in report.items()
        if k not in ("all_cases_summary", "tag_cooccurrence_top20")
    }, ensure_ascii=False, indent=2)
    sample = json.dumps(report["all_cases_summary"][:30], ensure_ascii=False, indent=2)

    # Call 1
    print("  [3a] Mental models + heuristics...")
    prompt1 = cfg["prompt_1"].format(
        total=report["total"], statistics=stats_str, sample_cases=sample
    )
    msg1 = client.messages.create(model=model, max_tokens=8000,
                                   messages=[{"role": "user", "content": prompt1}])
    text1 = msg1.content[0].text
    (data_dir / "llm_raw_response_1.txt").write_text(text1, encoding="utf-8")
    part1 = _parse_json(text1, "3a")

    # Call 2
    print("  [3b] Archetypes + anti-patterns...")
    sep = cfg["tpl"].get("vocab_sep", "、")
    top_tags = sep.join(list(report["top_personality_tags"].keys())[:10])
    cp = sep.join(f"{k}({v['count']})" for k, v in report["conflict_patterns"].items() if v["count"] > 5)
    sample2 = json.dumps(report["all_cases_summary"][:20], ensure_ascii=False, indent=2)
    prompt2 = cfg["prompt_2"].format(
        total=report["total"],
        male=report["gender"].get("male", 0),
        female=report["gender"].get("female", 0),
        age_min=report["age_stats"]["min"],
        age_max=report["age_stats"]["max"],
        age_avg=report["age_stats"]["avg"],
        top_tags=top_tags, conflict_patterns=cp, sample_cases=sample2
    )
    msg2 = client.messages.create(model=model, max_tokens=8000,
                                   messages=[{"role": "user", "content": prompt2}])
    text2 = msg2.content[0].text
    (data_dir / "llm_raw_response_2.txt").write_text(text2, encoding="utf-8")
    part2 = _parse_json(text2, "3b")

    frameworks = {**part1, **part2}
    out_path = data_dir / "extracted_frameworks.json"
    out_path.write_text(json.dumps(frameworks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Done → {out_path}")
    return frameworks


def _parse_json(text: str, label: str) -> dict:
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [{label}] JSON parse failed: {e}")
        print(f"  First 200 chars: {text[:200]}")
        sys.exit(1)


# ============================================================
# Step 3b: Case Digest (De-identification)
# ============================================================

def step3b_digest(cases: list, api_key: str, data_dir: Path, cfg: dict) -> str:
    print("[Step 3b] Case digest (de-identification)...")
    try:
        import anthropic
    except ImportError:
        print("  Error: pip install anthropic")
        sys.exit(1)

    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**kwargs)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    batch_size = 40
    all_digest = []

    for i in range(0, len(cases), batch_size):
        batch = cases[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(cases) + batch_size - 1) // batch_size
        print(f"  [3b-{batch_num}/{total_batches}] Processing {len(batch)} cases...")

        compact = []
        for c in batch:
            compact.append({
                "case_id": c["metadata"]["case_id"],
                "gender": c["subject_profile"]["gender"],
                "age": c["subject_profile"]["age"],
                "city": c["subject_profile"]["location"].get("city", ""),
                "tier": c["subject_profile"]["location"].get("tier_normalized", ""),
                "education": c["subject_profile"].get("education", ""),
                "career": c["subject_profile"].get("career", {}).get("primary", ""),
                "income": c["subject_profile"]["financials"].get("income_normalized", ""),
                "height": c["subject_profile"].get("physicals", {}).get("height_cm"),
                "personality": c["psychology_and_traits"]["personality_tags"],
                "core_conflict": c["matchmaking_intelligence"]["core_conflict"],
                "strategy": c["matchmaking_intelligence"]["expert_strategy"],
            })

        prompt = cfg["digest_prompt"].format(
            cases_json=json.dumps(compact, ensure_ascii=False, indent=2),
            first_id=compact[0]["case_id"]
        )

        msg = client.messages.create(
            model=model, max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )
        all_digest.append(msg.content[0].text)

    digest_md = "\n\n".join(all_digest)
    (data_dir / "case_digest_raw.md").write_text(digest_md, encoding="utf-8")
    print(f"  Digest done → {data_dir / 'case_digest_raw.md'}")
    return digest_md


# ============================================================
# Step 4: Skill Assembly
# ============================================================

def step4_assemble(frameworks: dict, report: dict, cases: list, output_dir: Path, cfg: dict, digest_md: str = "") -> str:
    print("[Step 4] Assembling SKILL.md...")
    t = cfg["tpl"]
    sep = t.get("vocab_sep", "、")

    models_md = ""
    for i, m in enumerate(frameworks["mental_models"], 1):
        evidence = "\n".join(f"  - {e}" for e in m["evidence"])
        models_md += f"""
### {t["model_heading"].format(i=i, name=m["name"])}
**{t["one_liner"]}**：{m["one_liner"]}
**{t["evidence"]}**：
{evidence}
**{t["application"]}**：{m["application"]}
**{t["limitation"]}**：{m["limitation"]}
"""

    heuristics_md = ""
    for i, h in enumerate(frameworks["heuristics"], 1):
        heuristics_md += f"""
{i}. **{h["rule"]}**：{h["description"]}
   - {t["scenario"]}：{h["scenario"]}
   - {t["case_example"]}：{h["case_example"]}
"""

    archetypes_md = ""
    for a in frameworks["archetypes"]:
        gender_label = "♂" if a["gender"] == "male" else "♀"
        archetypes_md += f"""
### {gender_label} {a["name"]}
**{t["profile"]}**：{a["profile"]}
**{t["core_conflict"]}**：{a["core_conflict"]}
**{t["strategy"]}**：{a["typical_strategy"]}
"""

    anti_md = ""
    for i, ap in enumerate(frameworks["anti_patterns"], 1):
        anti_md += f"""
{i}. **{ap["name"]}**（{t["frequency_label"]}：{ap["frequency"]}）
   - {ap["description"]}
   - {t["remedy"]}：{ap["remedy"]}
"""

    dna = frameworks["expression_dna"]
    vocab = sep.join(dna["vocabulary"][:10])
    taboo = sep.join(dna["taboo"][:5])

    male_n = report["gender"].get("male", 0)
    female_n = report["gender"].get("female", 0)
    sample_bias = t["sample_bias"].format(female=female_n, male=male_n)
    boundary_items = "\n".join(f"- {item}" for item in t["boundary_items"])
    boundary_items += f"\n- {sample_bias}"

    city_tier = report.get("city_tier", {})
    tier_parts = []
    for k, v in list(city_tier.items())[:5]:
        tier_parts.append(f"{k} {v}")
    city_dist = sep.join(tier_parts)

    skill_md = f"""---
name: {cfg["skill_name"]}
description: |
  {t["title"]}. Based on {report["total"]} real matchmaking cases,
  distilled {len(frameworks["mental_models"])} mental models, {len(frameworks["heuristics"])} heuristics,
  {len(frameworks["archetypes"])} archetypes and {len(frameworks["anti_patterns"])} anti-patterns.
  {t["trigger_desc"]}
---

# {t["title"]}

> {t["quote"]}

## {t["usage_title"]}

{t["usage_body"]}

## {t["workflow_title"]}

{t["workflow_intro"]}

### {t["step1_title"]}

{t["step1_body"]}

### {t["step2_title"]}

{t["step2_body"]}

### {t["step3_title"]}

{t["step3_body"]}

### {t["step4_title"]}

{t["step4_body"]}

### {t["step5_title"]}

{t["step5_body"]}

---

## {t["models_title"]}
{models_md}

---

## {t["heuristics_title"]}
{heuristics_md}

---

## {t["archetypes_title"]}
{archetypes_md}

---

## {t["anti_title"]}
{anti_md}

---

## {t["dna_title"]}

{t["dna_intro"]}
- **{t["dna_style"]}**：{dna["style"]}
- **{t["dna_sentence"]}**：{dna["sentence_pattern"]}
- **{t["dna_vocab"]}**：{vocab}
- **{t["dna_tone"]}**：{dna["tone"]}
- **{t["dna_taboo"]}**：{taboo}

---

## {t["boundary_title"]}

{boundary_items}

## {t["appendix_title"]}

- Platform: {t["appendix_platform"]}
- Blogger: {t["appendix_blogger"]}
- Population: {report["age_stats"]["min"]}-{report["age_stats"]["max"]}, {male_n}M/{female_n}F
- City distribution: {city_dist}

---

> {t["footer"]}
"""

    out_path = output_dir / "SKILL.md"
    out_path.write_text(skill_md, encoding="utf-8")
    print(f"  SKILL.md → {out_path}")

    # --- Enhanced References ---

    # 1. statistics-snapshot.md (pure Python)
    _gen_statistics_snapshot(report, output_dir, t)

    # 2. pattern-library.md (enhanced with case details)
    _gen_pattern_library(report, cases, output_dir, t)

    # 3. case-archive.md (full table, all cases)
    archive_md = f"# {t['archive_title']}\n\n{t['archive_headers']}\n|---------|------|------|------|----------|\n"
    for c in report.get("all_cases_summary", []):
        archive_md += f"| {c['case_id']} | {c['gender']} | {c['age']} | {c['income']} | {c['core_conflict'][:40]}... |\n"
    (output_dir / "references" / "case_archive.md").write_text(archive_md, encoding="utf-8")

    # 4. case-digest.md (LLM de-identified summaries)
    if digest_md:
        header = f"# {t['digest_title']}\n\n> {t['digest_note']}\n\n---\n\n"
        (output_dir / "references" / "case_digest.md").write_text(header + digest_md, encoding="utf-8")

    print("  References done")
    return skill_md


def _gen_statistics_snapshot(report: dict, output_dir: Path, t: dict):
    sep = t.get("vocab_sep", ", ")
    lines = [f"# {t['stats_title']}", "", f"> {t['stats_generated'].format(total=report['total'])}", ""]

    # Gender
    lines.append(f"## {t['stats_gender']}")
    for k, v in report["gender"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Age
    a = report["age_stats"]
    lines.append(f"## {t['stats_age']}")
    lines.append(f"- Min: {a['min']}, Max: {a['max']}, Avg: {a['avg']}")
    lines.append("")

    # City tier
    lines.append(f"## {t['stats_city']}")
    for k, v in report["city_tier"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Income
    lines.append(f"## {t['stats_income']}")
    for k, v in report["income"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Education
    lines.append(f"## {t['stats_edu']}")
    for k, v in report["education"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Tags
    tags = report["top_personality_tags"]
    lines.append(f"## {t['stats_tags_title'].format(n=len(tags))}")
    lines.append("")
    lines.append("| Tag | Count |")
    lines.append("|-----|-------|")
    for tag, cnt in tags.items():
        lines.append(f"| {tag} | {cnt} |")
    lines.append("")

    # Male/Female tags
    for key, label_key in [("male_top_tags", "stats_male_tags"), ("female_top_tags", "stats_female_tags")]:
        data = report.get(key, {})
        lines.append(f"## {t[label_key].format(n=len(data))}")
        lines.append("")
        lines.append("| Tag | Count |")
        lines.append("|-----|-------|")
        for tag, cnt in data.items():
            lines.append(f"| {tag} | {cnt} |")
        lines.append("")

    # Co-occurrence
    cooc = report.get("tag_cooccurrence_top20", [])
    lines.append(f"## {t['stats_cooccur'].format(n=len(cooc))}")
    lines.append("")
    lines.append("| Pair | Count |")
    lines.append("|------|-------|")
    for item in cooc:
        lines.append(f"| {sep.join(item['pair'])} | {item['count']} |")
    lines.append("")

    # Conflict keywords
    ckw = report.get("conflict_keywords_top30", {})
    lines.append(f"## {t['stats_conflict_kw'].format(n=len(ckw))}")
    lines.append("")
    lines.append("| Keyword | Count |")
    lines.append("|---------|-------|")
    for kw, cnt in ckw.items():
        lines.append(f"| {kw} | {cnt} |")
    lines.append("")

    # Strategy keywords
    skw = report.get("strategy_keywords_top30", {})
    lines.append(f"## {t['stats_strategy_kw'].format(n=len(skw))}")
    lines.append("")
    lines.append("| Keyword | Count |")
    lines.append("|---------|-------|")
    for kw, cnt in skw.items():
        lines.append(f"| {kw} | {cnt} |")
    lines.append("")

    # Conflict pattern distribution
    lines.append(f"## {t['stats_pattern_dist']}")
    lines.append("")
    lines.append("| Pattern | Count |")
    lines.append("|---------|-------|")
    for pattern, info in report.get("conflict_patterns", {}).items():
        lines.append(f"| {pattern} | {info['count']} |")
    lines.append("")

    (output_dir / "references" / "statistics_snapshot.md").write_text("\n".join(lines), encoding="utf-8")


def _gen_pattern_library(report: dict, cases: list, output_dir: Path, t: dict):
    case_map = {}
    for s in report.get("all_cases_summary", []):
        case_map[s["case_id"]] = s

    lines = [f"# {t['pattern_lib_title']}", ""]
    for pattern, info in report.get("conflict_patterns", {}).items():
        lines.append(f"## {pattern}（{info['count']}）")
        lines.append("")
        lines.append(f"**{t['pattern_typical']}**：")
        lines.append("")
        for cid in info["cases"]:
            s = case_map.get(cid)
            if s:
                lines.append(f"- **{cid}**（{s['gender']}, {s['age']}, {s['income']}）：{s['core_conflict']}")
                lines.append(f"  → {s['strategy']}")
        lines.append("")

    (output_dir / "references" / "pattern_library.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Step 5: Quality Check
# ============================================================

def step5_quality_check(skill_path: Path, output_dir: Path, cfg: dict) -> bool:
    print("[Step 5] Quality check...")
    content = skill_path.read_text(encoding="utf-8")
    qc = cfg["qc"]
    results = []

    models = re.findall(qc["model_re"], content, re.MULTILINE)
    count = len(models)
    ok = 3 <= count <= 7
    results.append(("Mental Models", ok, f"{count} {'✅' if ok else '❌ (need 3-7)'}"))

    has_limit = bool(re.search(qc["limit_re"], content, re.IGNORECASE))
    results.append(("Limitations", has_limit, "✅" if has_limit else "❌"))

    heuristics = re.findall(r'^\d+\.\s+\*\*', content, re.MULTILINE)
    h_count = len(heuristics)
    ok = h_count >= 8
    results.append(("Heuristics", ok, f"{h_count} {'✅' if ok else '❌ (need ≥8)'}"))

    archetypes = re.findall(qc["archetype_re"], content, re.MULTILINE)
    a_count = len(archetypes)
    ok = a_count >= 6
    results.append(("Archetypes", ok, f"{a_count} {'✅' if ok else '❌ (need ≥6)'}"))

    boundary = re.search(qc["boundary_section_re"], content, re.DOTALL)
    b_items = len(re.findall(r'^-\s+', boundary.group(1), re.MULTILINE)) if boundary else 0
    ok = b_items >= 3
    results.append(("Boundaries", ok, f"{b_items} {'✅' if ok else '❌ (need ≥3)'}"))

    anti = re.findall(qc["anti_re"], content, re.MULTILINE | re.IGNORECASE)
    ok = len(anti) >= 3
    results.append(("Anti-patterns", ok, f"{len(anti)} {'✅' if ok else '❌ (need ≥3)'}"))

    report_lines = ["Quality Report", "=" * 40]
    passed = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        report_lines.append(f"  {name:<16} {status}  {detail}")
        if ok:
            passed += 1
    report_lines.append("=" * 40)
    report_lines.append(f"Result: {passed}/{len(results)} passed")
    report_lines.append("All passed ✅" if passed == len(results) else "Has failures ⚠️")

    report_text = "\n".join(report_lines)
    print(report_text)
    (output_dir / "quality_report.txt").write_text(report_text, encoding="utf-8")
    return passed == len(results)


# ============================================================
# Main
# ============================================================

def run_pipeline(lang: str, skip_llm: bool, api_key=None):
    cfg = I18N[lang]
    suffix = cfg["output_suffix"]
    data_dir = BASE_DIR / "data" / suffix
    output_dir = BASE_DIR / suffix
    data_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "references").mkdir(parents=True, exist_ok=True)

    source_path = DATASETS_DIR / cfg["source_file"]
    if not source_path.exists():
        print(f"Error: source not found: {source_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Pipeline [{lang.upper()}] — {source_path.name}")
    print(f"{'='*50}\n")

    cases = step1_clean(source_path, data_dir, cfg)
    report = step2_analyze(cases, data_dir, cfg)

    if skip_llm:
        print(f"\n[Skip Step 3-4] --skip-llm")
        print(f"Stats: {data_dir / 'statistics_report.json'}")
        return

    if not api_key:
        print("Error: need ANTHROPIC_API_KEY (--api-key or env var)")
        sys.exit(1)

    frameworks = step3_extract(cases, report, api_key, data_dir, cfg)
    digest_md = step3b_digest(cases, api_key, data_dir, cfg)
    step4_assemble(frameworks, report, cases, output_dir, cfg, digest_md=digest_md)
    step5_quality_check(output_dir / "SKILL.md", output_dir, cfg)

    print(f"\n✅ [{lang.upper()}] Done! → {output_dir / 'SKILL.md'}")


def main():
    parser = argparse.ArgumentParser(description="Matchmaking Data Distill Pipeline")
    parser.add_argument("--lang", default="zh", choices=["zh", "en", "all"],
                        help="Language: zh (Chinese), en (English), all (both)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM extraction")
    parser.add_argument("--api-key", help="Anthropic API Key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")

    langs = ["zh", "en"] if args.lang == "all" else [args.lang]
    for lang in langs:
        run_pipeline(lang, args.skip_llm, api_key)

    print("\n🎉 All pipelines complete!")


if __name__ == "__main__":
    main()
