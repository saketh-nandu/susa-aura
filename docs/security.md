# SUSA AURA — Security & Safety Architecture

> **Adaptive Universal Reasoning Assistant (SUSA AURA)**
> Defense-in-depth security model for autonomous assistance, command execution, and internet retrieval.

---

## 1. Core Security Principles

1. **Explicit Authorization Gates:** Destructive file modifications, non-allowlisted command execution, and persisting private information always require explicit user approval (`user_approved=True`).
2. **Untrusted Web Content Is Data, Not Instructions:** Retrieved web pages and external documentation (via AURA Connect) are treated strictly as untrusted evidence context. Prompt injection patterns (such as "ignore all previous instructions") are neutralized before prompt synthesis.
3. **Automated Secret Scrubbing:** Credentials, API keys, JWTs, AWS credentials, and private keys are redacted on ingestion, on memory write, and prior to logging or prompt insertion via `aura.data.cleaner.DataCleaner`.
4. **Local-Only Mode & Privacy Enforcement:** When `AURA_LOCAL_ONLY=1` (the default), external network calls to third-party model providers are strictly blocked with `PermissionError`. No user prompts, code files, or memories are exfiltrated.
5. **Path Traversal Protection:** File reading and writing operations canonicalize paths (`Path.resolve()`) and verify that target paths remain strictly confined within the workspace root.
6. **Automatic Reversible Backups:** Any modified file generates a local `.bak` backup before writing to disk.
7. **Execution Budgets & Loop Detection:** Bounded execution plans enforce strict step limits (default 8) and detect repeated identical tool calls without state progression (`LoopDetectedError`), halting execution before resource exhaustion.

---

## 2. Command Allowlisting & Tool Parameter Validation

The AURA Tools safe command runner (`aura.tools.command_runner.SafeCommandRunner`) enforces an allowlist for automated test executions:
- `pytest`, `python -m pytest`
- `python scripts/verify_env.py`
- `python scripts/count_params.py`
- `python scripts/evaluate.py`
- `git status`, `git diff`, `git log`
- `ls`, `pwd`

Commands outside the allowlist are rejected unless the user explicitly authorizes the action.
All tool calls undergo typed schema validation (`ToolParameter`) before execution, rejecting missing or invalid parameter types.

---

## 3. Sandboxing & Remaining Limitations

- **Process Isolation:** Ephemeral Python code evaluation operates in isolated subprocesses (`python -I -s`) with stripped environment variables, CPU execution timeouts (3-5s), and output truncation caps (8,000 chars).
- **Security Boundary Realism:** Process isolation via isolated subprocesses is not equivalent to a kernel-level container or microVM. For executing completely untrusted foreign code from the internet, users should run AURA within a Linux container (Docker/Podman) or virtualized sandbox.

---

## 4. Environment Variables & Secret Hygiene

- **Never Commit Secrets:** `.env` and `.env.local` are explicitly ignored by Git.
- **Prefix:** Internal configuration uses `AURA_*` environment variables (`AURA_DATA_DIR`, `AURA_CHECKPOINT_DIR`, `AURA_LOCAL_ONLY`, etc.).
- **No Secret Echoing:** Logs, audit records, and error traces redact sensitive credentials.
