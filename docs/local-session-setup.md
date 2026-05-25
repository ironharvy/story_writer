# Local `/goal` session setup (WSL + native sandbox + Ollama)

How to run the story-generator `/goal` brief on a **local** Claude Code instance,
sandboxed with bubblewrap, iterating against Ollama. Pairs with
`STORY_EVAL_SPEC.md` (the objective function) and `STORY_GOAL_PROMPT.md` (the
brief).

The model running the local session is in **auto** permission mode: an LLM
classifier judges each action and only stops to ask on the genuinely risky ones.
The sandbox is the safety net underneath it.

---

## 1. Pre-flight (one-time, run on the host — NOT inside a sandboxed session)

```bash
# Sandbox dependencies (Debian/Ubuntu WSL)
sudo apt install -y bubblewrap socat

# Ollama up, with the models pulled. Do this on the host: `ollama pull` writes to
# ~/.ollama, which is OUTSIDE the working dir and would be blocked under the sandbox.
ollama serve                     # if not already running
ollama pull qwen3:latest         # fast dev-loop model
ollama pull qwen3.6:27b          # quality model
ollama pull nemotron-3-nano      # optional: to benchmark as an alt backend / judge

# In the greenfield repo: a project-local venv so `pip install` writes stay INSIDE
# the working dir (the sandbox confines writes there by default).
cd <greenfield-repo>
python -m venv .venv && source .venv/bin/activate

# DeepSeek paid-fallback key (only used when local models can't clear the eval bar).
export DEEPSEEK_API_KEY=sk-...

# Keep the DSPy/LLM cache inside the working dir too, so cache writes aren't blocked.
export DSPY_CACHE_DIR="$PWD/.cache/dspy"
```

**Restart Claude Code after installing bubblewrap/socat** — the sandbox
dependency check only runs at startup, so `/sandbox` won't detect them until you
relaunch.

Then copy `STORY_EVAL_SPEC.md` (and `STORY_GOAL_PROMPT.md` for reference) into
the repo root.

## 2. `.claude/settings.json`

Create this in the greenfield repo. (Prefer `.claude/settings.local.json` if you
don't want to commit machine-specific config — same shape, gitignored.)

```json
{
  "permissions": {
    "defaultMode": "auto",
    "allow": [
      "Bash(git status)",
      "Bash(git diff *)",
      "Bash(git log *)",
      "Bash(git add *)",
      "Bash(git commit *)",
      "Bash(pytest *)",
      "Bash(python *)",
      "Bash(pip install *)",
      "Bash(ollama *)"
    ]
  },
  "sandbox": {
    "enabled": true,
    "network": {
      "allowLocalBinding": true,
      "allowedDomains": [
        "localhost",
        "127.0.0.1",
        "pypi.org",
        "files.pythonhosted.org",
        "registry.ollama.ai",
        "ollama.com",
        "api.deepseek.com",
        "github.com"
      ]
    }
  }
}
```

What each part does:

- **`permissions.defaultMode: "auto"`** — the LLM classifier decides per-action
  whether to auto-run or ask. The first time you use auto mode Claude Code shows a
  one-time opt-in dialog; accept it (for a fully unattended start you can pre-accept
  by adding `"skipAutoPermissionPrompt": true` at the top level, but only do that
  knowing it records consent).
- **`permissions.allow`** — optional pre-approvals for obviously-safe commands, so
  they skip the classifier (less latency/cost). Delete the array to route
  *everything* through the LLM judge. Note `git push` is deliberately **not** here —
  let auto mode pause on shared-state actions.
- **`sandbox.enabled: true`** — bubblewrap FS isolation + socat network proxy.
  Left as warn-and-continue (no `failIfUnavailable`), so if the sandbox can't start
  it runs unsandboxed with a warning rather than refusing.
- **`sandbox.network.allowedDomains`** — the allowlist for **Bash-run** network
  access only. `localhost`/`127.0.0.1` for Ollama; PyPI for `pip`; the Ollama
  registry in case of an in-session pull; `api.deepseek.com` for the paid fallback;
  `github.com` only if you'll push (drop it otherwise). Claude's own
  `WebSearch`/`WebFetch` research runs outside the sandbox, so the craft-research
  step is **not** gated by this list.

## 3. Verify Ollama is reachable from inside the sandbox (the one thing to check)

On Linux the sandbox runs Bash in its own network namespace, so a sandboxed
process reaching the host's `localhost:11434` is the most likely thing to need a
tweak. First action in the session, have Claude run:

```bash
curl -s http://localhost:11434/api/tags
```

If it returns the model list, the pipeline can reach Ollama — you're good.

If it **fails**, the reliable fallback is to run the generation command outside
the sandbox (it's your own code; pip/installs and everything else stay
sandboxed). Add to the `sandbox` block:

```json
"excludedCommands": ["python"]
```

(or scope it to the specific run script). That command then gets normal host
networking and reaches Ollama directly.

## 4. Launch the run

Start Claude Code in the repo, confirm sandbox + auto mode are active (`/sandbox`,
`/status`), run the Ollama check above, then paste the `/goal` prompt from
`STORY_GOAL_PROMPT.md`. It will ask clarifying questions before generating.
