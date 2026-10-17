# SUSA AURA — Dataset Provenance & Data Policy

> **Adaptive Universal Reasoning Assistant (SUSA AURA)**
> Standards for licensed, high-quality, sanitized data ingestion.

---

## 1. Compliance and Permitted Licenses

All data ingested into the pretraining, retrieval, and fine-tuning pipelines of SUSA AURA must strictly originate from permitted open-source licenses:
- MIT
- Apache 2.0
- BSD-2-Clause / BSD-3-Clause
- ISC
- Unlicense / Public Domain (CC0)
- Python Software Foundation (PSF) License (Python Standard Library)

**Prohibited Data Sources:**
- Copyleft-restricted code (GPL-3.0, AGPL-3.0) without explicit compatibility vetting.
- Proprietary or internal confidential codebase leaks.
- Non-commercial only licenses (CC-BY-NC) if general-purpose deployment is targeted.
- Raw web scrapes without documented provenance or robots.txt compliance.

---

## 2. PII and Secret Filtering (Non-Negotiable)

Before tokenization or indexing, all raw text and code samples undergo automated multi-pass regex and heuristic sanitization (`aura.data.cleaner.DataCleaner`):
1. **API Keys & Tokens:** GitHub Personal Access Tokens (`ghp_`, `github_pat_`), AWS Access Keys (`AKIA...`), OpenAI/Anthropic/HuggingFace API keys.
2. **Private Keys & Certificates:** RSA/SSH Private keys (`-----BEGIN RSA PRIVATE KEY-----`).
3. **Database Credentials & Passwords:** Hardcoded connection strings, passwords, and tokens.
4. **Emails & Credentials:** Stripped or masked to prevent personal data exposure.

---

## 3. Deduplication and Quality Filtering

- **Exact & MinHash Deduplication:** Deduplicate files across corpora using exact SHA-256 hashing and MinHash Jaccard similarity (k=4 shingles, 32 permutations) to prevent repetition of boilerplate.
- **Line Length and File Size Filters:** Remove files with average line length > 1,000 characters (minified code) or files > 1 MB.
- **Empty / Malformed Filtering:** Exclude empty files, syntax error files, or non-text binaries.
- **Zero Test-Set Contamination:** Never train tokenizer or model on benchmark test sets (HumanEval, MBPP, or custom evaluation assertions).
