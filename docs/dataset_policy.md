# SUSA AURA — Dataset Provenance & Safety Policy

## 1. Compliance and Permitted Licenses
All data ingested into the pretraining and instruction-tuning pipelines of SUSA-1B CodeMind must strictly originate from permitted open-source licenses:
- MIT
- Apache 2.0
- BSD-2-Clause / BSD-3-Clause
- ISC
- Unlicense / Public Domain (CC0)

**Prohibited Data Sources:**
- Copyleft-restricted code (GPL-3.0, AGPL-3.0) unless specifically vetted.
- Proprietary or internal codebase leaks.
- Non-commercial only licenses (CC-BY-NC) if general-purpose reuse is desired.
- Data gathered without documented provenance.

---

## 2. PII and Secret Filtering (Non-Negotiable)
Before tokenization, all raw code samples must undergo regex and heuristic scanning to strip:
1. **API Keys & Tokens:** GitHub Personal Access Tokens (`ghp_`), AWS Access Keys (`AKIA...`), OpenAI/Anthropic/HuggingFace API keys.
2. **Private Keys & Certificates:** RSA/SSH Private keys (`-----BEGIN RSA PRIVATE KEY-----`).
3. **Database Credentials & Passwords:** Hardcoded connection strings, passwords, and secrets.
4. **Emails & Phone Numbers:** Stripped or masked to prevent personal data exposure.

---

## 3. De-duplication and Quality Filtering
- **MinHash / Exact Deduplication:** Deduplicate files across repositories to prevent over-representation of common boilerplate (e.g., license headers, vendored packages).
- **Line Length and File Size Filters:** Remove files with average line length > 1,000 characters (minified code) or files > 1 MB.
- **Empty / Malformed Filtering:** Exclude empty files, syntax error files, or non-text binaries.
- **No Test-Set Contamination:** Never train the tokenizer or model on evaluation/benchmark test sets (e.g., HumanEval, MBPP, custom test suites).
