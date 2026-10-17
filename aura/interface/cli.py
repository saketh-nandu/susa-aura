"""
SUSA AURA — Interactive Command-Line Interface (AURA CLI).

Provides a rich conversational REPL distinguishing:
- Model-generated reasoning and code completions.
- Retrieved external research sources and URLs (AURA Connect).
- Tool calls, unified diffs, and workspace inspection (AURA Tools).
- Retrieved memory preferences and facts (AURA Memory).
- Transparent error and limitation reporting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aura import __version__
from aura.core.runtime import AuraCore
from aura.inference.generator import AuraGenerator
from aura.model.config import ModelConfig


def print_banner() -> None:
    banner = f"""
================================================================================
  ✦ SUSA AURA — Adaptive Universal Reasoning Assistant (v{__version__}) ✦
  Architecture: Decoder-Only Transformer (RoPE, SwiGLU, RMSNorm, GQA)
================================================================================
  Type '/help' for commands, '/search <query>' for web research,
  '/inspect' to view workspace, '/diff' to propose edits, or 'exit' to quit.
================================================================================
"""
    print(banner)


def print_help() -> None:
    help_text = """
Commands:
  /help                  Show this help menu
  /inspect               Display workspace file tree
  /search <query>        Execute research retrieval via AURA Connect
  /memory                List active memories and user preferences
  /remember <key> <val>  Save a persistent fact or preference to memory
  /forget <key>          Remove a fact from memory
  /diff <file> <text>    Preview unified diff without mutating disk
  /tools                 List all registered AURA tools
  /stats                 Show session statistics and action log
  exit, quit             Exit SUSA AURA CLI
"""
    print(help_text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SUSA AURA: Adaptive Universal Reasoning Assistant CLI",
        prog="aura",
    )
    parser.add_argument("--workspace", type=str, default=".", help="Project workspace root directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model weights checkpoint (.pt).")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer_cache/smoke", help="Path to tokenizer directory.")
    parser.add_argument("--config", type=str, default="configs/smoke.yaml", help="Path to model config YAML.")
    parser.add_argument("--offline", action="store_true", help="Disable online search retrieval.")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt mode (single-turn).")
    parser.add_argument("--inspect", action="store_true", help="Inspect workspace and exit.")

    args = parser.parse_args()

    # Load generator if checkpoint is provided
    generator = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        tok_path = Path(args.tokenizer_dir)
        if ckpt_path.exists() and tok_path.exists():
            try:
                print(f"[*] Loading AURA model from: {ckpt_path}")
                generator = AuraGenerator.from_checkpoint(
                    checkpoint_path=ckpt_path,
                    tokenizer_path=tok_path,
                    config_path=args.config if Path(args.config).exists() else None,
                )
                print(f"[✓] Model loaded ({generator.model.count_parameters():,} params).")
            except Exception as e:
                print(f"[!] Warning: Could not load checkpoint ({e}). Running in core diagnostic mode.")

    core = AuraCore(workspace_root=args.workspace, generator=generator)
    if args.offline:
        core.session.research_enabled = False

    if args.inspect:
        print(core.inspect_workspace())
        return 0

    if args.prompt:
        result = core.ask(args.prompt, enable_search=not args.offline)
        print("\n--- [AURA RESPONSE] ---")
        print(result["response"])
        if result["sources"]:
            print("\n--- [GROUNDED SOURCES] ---")
            for s in result["sources"]:
                print(f"  • {s.title} ({s.url})")
        return 0

    # Interactive REPL
    print_banner()

    while True:
        try:
            user_input = input("AURA> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye from SUSA AURA!")
                break

            if user_input == "/help":
                print_help()
                continue

            if user_input == "/inspect":
                print("\n[WORKSPACE TREE]:")
                print(core.inspect_workspace())
                continue

            if user_input == "/tools":
                print("\n[REGISTERED AURA TOOLS]:")
                for name, defn in core.tools.tools.items():
                    print(f"  • {name}: {defn.description} (Requires Approval: {defn.requires_approval})")
                continue

            if user_input == "/memory":
                print("\n[ACTIVE MEMORIES]:")
                records = core.memory.list_all()
                if not records:
                    print("  (No persistent memories saved yet.)")
                for r in records:
                    print(f"  • [{r.category}] {r.key}: {r.content}")
                continue

            if user_input.startswith("/remember "):
                parts = user_input[10:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    k, v = parts
                    core.memory.store_fact(key=k, content=v, user_approved=True)
                    print(f"[✓] Stored in AURA memory: '{k}'")
                else:
                    print("[!] Usage: /remember <key> <fact or preference>")
                continue

            if user_input.startswith("/forget "):
                k = user_input[8:].strip()
                if core.memory.delete_fact(k):
                    print(f"[✓] Forgotten from AURA memory: '{k}'")
                else:
                    print(f"[!] Key '{k}' not found in memory.")
                continue

            if user_input == "/stats":
                print("\n" + core.summary())
                continue

            if user_input.startswith("/search "):
                q = user_input[8:].strip()
                print(f"[*] Researching via AURA Connect: '{q}'...")
                res = core.connect.research(q)
                if res.sources:
                    print("\n[RETRIEVED SOURCES]:")
                    for s in res.sources:
                        print(f"  • {s.title}: {s.snippet[:120]}...\n    URL: {s.url}")
                else:
                    print(f"  {res.context_text or 'No sources found.'}")
                continue

            # Standard dialogue turn
            enable_search = not args.offline and any(
                term in user_input.lower() for term in ["search", "find", "research", "what is", "how to", "latest", "doc"]
            )
            res = core.ask(user_input, enable_search=enable_search)

            if res["sources"]:
                print("\n--- [SOURCES RETRIEVED (AURA CONNECT)] ---")
                for s in res["sources"]:
                    print(f"  • {s.title} ({s.url})")

            print("\n--- [AURA RESPONSE] ---")
            print(res["response"])
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting SUSA AURA.")
            break
        except Exception as err:
            print(f"\n[!] Error during turn: {err}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
