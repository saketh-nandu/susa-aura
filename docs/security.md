# SUSA AURA — Security & Safety Architecture

> **Adaptive Universal Reasoning Assistant (SUSA AURA)**
> Defense-in-depth security model for autonomous assistance, command execution, and internet retrieval.

---

## 1. Core Security Principles

1. **Explicit Authorization Gates:** Destructive file modifications, arbitrary command execution, and persisting private information always require explicit user approval (`user_approved=True`).
2. **Untrusted Web Content Is Data, Not Instructions:** Retrieved web pages and external documentation (via AURA Connect) are treated strictly as untrusted evidence context. Prompt injection patterns (such as "ignore all previous instructions") are neutralized before prompt synthesis.
3. **Automated Secret Scrubbing:** Credentials, API keys, JWTs, AWS credentials, and private keys are redacted both on ingestion and prior to logging or prompt insertion.
4. **Isolated Sandboxing:** Untrusted code execution operates in isolated ephemeral subprocesses (`aura.evaluation.sandbox.IsolatedCodeRunner`) with CPU timeouts, memory caps, stripped environments (`-I -s`), and zero access to workspace root secrets.
5. **Path Traversal Protection:** File reading and writing operations verify canonical path resolution to prevent directory traversal (`../`) outside the authorized workspace root.
6. **Automatic Reversible Backups:** Any modified file generates a local `.bak` backup before writing to disk.

---

## 2. Command Allowlisting

The AURA Tools safe command runner (`aura.tools.command_runner.SafeCommandRunner`) enforces an allowlist for automated test executions:
- `pytest`, `python -m pytest`
- `python scripts/verify_env.py`
- `python scripts/count_params.py`
- `python scripts/evaluate.py`
- `git status`, `git diff`, `git log`
- `ls`, `pwd`

Commands outside the allowlist are rejected unless the user explicitly flags `--apply` or authorizes the action in the CLI.

---

## 3. Environment Variables & Secret Hygiene

- **Never Commit Secrets:** `.env` and `.env.local` are explicitly ignored by Git.
- **Prefix:** Internal configuration uses `AURA_*` environment variables (`AURA_DATA_DIR`, `AURA_CHECKPOINT_DIR`, etc.).
- **No Secret Echoing:** Logs and error traces avoid printing sensitive credentials.
