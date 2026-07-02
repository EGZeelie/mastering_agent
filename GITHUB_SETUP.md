# GitHub Repository Configuration

Copy-paste-ready settings for configuring this repo on GitHub, plus the
GitHub-native apps/features to enable. Everything referenced here
(`.github/workflows/`, `dependabot.yml`, issue templates, etc.) already
exists in this repo — this doc is the checklist for wiring it up in the
GitHub UI/settings after you push.

---

## 1. Repository description & metadata

**Repo name:** `mastering-agent`

**Short description** (appears under the repo name, used in search/listing —
GitHub limits this to ~350 characters):
```
Autonomous, closed-loop AI mastering agent. Analyzes a mix (loudness, tone, dynamics), reasons over target parameters via Gemini or a deterministic rule engine, and iteratively renders through a linear-phase EQ → dynamic EQ → compressor → saturation → true-peak limiter chain — with a hard crest-factor floor so it never over-limits.
```

**Website field:** leave blank, or link to internal docs/demo if you have one.

**Topics** (add via the gear icon next to "About" on the repo homepage —
these drive GitHub search/discovery):
```
audio-mastering
audio-processing
dsp
digital-signal-processing
music-production
ai-agent
llm
gemini
mcp
python
music-tech
loudness-normalization
audio-engineering
agentic-ai
```

**About section checkboxes to enable:**
- [x] Releases (even if you don't tag versions yet, useful later)
- [x] Packages (if you later publish to PyPI)
- [ ] Environments (not needed — no deployment target)
- [x] Use your repository description

---

## 2. Repository settings (Settings tab)

### General
- **Default branch:** `main`
- **Features:**
  - [x] Issues
  - [x] Discussions *(recommended — good place for architecture debate that
        isn't a bug/feature, replaces the placeholder link in
        `.github/ISSUE_TEMPLATE/config.yml`)*
  - [ ] Wiki (off — keep documentation in-repo as Markdown, already
        comprehensive via README/INSTALL/CLAUDE.md)
  - [x] Projects (optional, useful once you're tracking multi-issue work
        like "batch/album processing" from the README's known-limitations list)
- **Pull Requests:**
  - [x] Allow squash merging (recommended as the default/only option for a
        clean linear history)
  - [ ] Allow merge commits
  - [ ] Allow rebase merging
  - [x] Always suggest updating pull request branches
  - [x] Automatically delete head branches after merge

### Code security and analysis (Settings → Code security)
- [x] **Dependabot alerts** — on
- [x] **Dependabot security updates** — on
- [x] **Dependabot version updates** — on (reads `.github/dependabot.yml`,
      already committed)
- [x] **Secret scanning** — on
- [x] **Secret scanning push protection** — on (blocks commits containing
      things that look like API keys, e.g. an accidentally-committed
      `GEMINI_API_KEY`)
- [x] **CodeQL analysis** — on (workflow already committed at
      `.github/workflows/codeql.yml`; GitHub will detect and offer to use
      the existing workflow rather than the default setup)
- [x] **Private vulnerability reporting** — on (required for `SECURITY.md`'s
      reporting instructions to work)

### Branch protection (Settings → Branches → Add branch protection rule)

Rule for `main`:
- Branch name pattern: `main`
- [x] Require a pull request before merging
  - [x] Require approvals — **1** minimum
  - [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Required checks (will appear in the list after the first CI run on a
    PR — select all of these):
    - `Test (Python 3.10)`
    - `Test (Python 3.11)`
    - `Test (Python 3.12)`
    - `Lint (ruff)`
    - `Analyze (python)` *(CodeQL)*
- [x] Require conversation resolution before merging
- [x] Do not allow bypassing the above settings *(include administrators)*
- [ ] Allow force pushes — off
- [ ] Allow deletions — off

### Tags/Releases (optional, once you're ready to version)
- Use semantic versioning tags (`v0.1.0`, `v0.2.0`, ...) matching the
  `version` field in `pyproject.toml`.

---

## 3. Social preview image (optional but recommended for a public repo)

Settings → General → Social preview → upload a 1280×640px image. Since this
project has no logo yet, a simple option: a screenshot of the `run_demo.py`
convergence output, or a diagram of the Ears→Brain→Hands→Loop architecture
from the README.

---

## 4. GitHub Apps / integrations to install

None of these are strictly required beyond what's already wired via
workflows/Dependabot above, but recommended for a project at this stage:

| App | Purpose | Setup |
|---|---|---|
| **Dependabot** | Already configured via `.github/dependabot.yml` — just enable "Dependabot version updates" in repo settings (see above); no separate app install needed, it's native to GitHub. |
| **CodeQL** | Already configured via `.github/workflows/codeql.yml` — enable "CodeQL analysis" in repo settings (above); native to GitHub, no separate app install. |
| **Codecov** *(optional)* | Test coverage reporting/badges on PRs. Install from the [GitHub Marketplace](https://github.com/marketplace/codecov), then add a `CODECOV_TOKEN` repo secret and a step to `ci.yml`'s test job: `pytest --cov=mastering_agent --cov-report=xml` + `codecov/codecov-action@v4`. |
| **Pre-commit.ci** *(optional)* | Auto-runs `pre-commit` hooks (ruff, etc.) on every PR without needing local hook installation. Only useful if you adopt a `.pre-commit-config.yaml` — not included by default here since `ruff check .` in CI already covers linting. |

None of these require paid plans for a public repo at this scale.

---

## 5. Repository secrets (Settings → Secrets and variables → Actions)

Only needed if/when you want CI itself to exercise the real Gemini path
(current CI intentionally does **not** require this — it validates the
rule-based fallback, which is deterministic and doesn't burn API quota on
every push):

| Secret name | Used by | Required? |
|---|---|---|
| `GEMINI_API_KEY` | Not currently referenced by any workflow. Add only if you extend `ci.yml` with a job that specifically tests live Gemini calls. | No |
| `CODECOV_TOKEN` | Only if you add Codecov (see above). | No |

---

## 6. Labels (Issues → Labels)

The issue templates reference these labels — create them (Settings → or
Issues tab → Labels → New label) if they don't already exist by default:

| Label | Color | Used by |
|---|---|---|
| `bug` | `#d73a4a` | default GitHub label, already exists |
| `triage` | `#fbca04` | `bug_report.yml` |
| `enhancement` | default GitHub label, already exists | `feature_request.yml` |
| `dependencies` | `#0366d6` | `dependabot.yml` |
| `python` | `#3572A5` | `dependabot.yml` |
| `ci` | `#000000` | `dependabot.yml` |

---

## 7. First-push checklist

```bash
cd mastering_agent
git init
git add .
git commit -m "Initial commit: autonomous mastering agent prototype"
git branch -M main
git remote add origin https://github.com/OWNER/mastering-agent.git
git push -u origin main
```

Then, in order:
1. Go through **Section 1** (description, topics, About settings).
2. Go through **Section 2** (features, PR settings, code security toggles).
3. Open a throwaway PR (e.g. a whitespace fix) to trigger the first CI +
   CodeQL run — the status checks won't be selectable in branch protection
   until they've run at least once.
4. Come back and finish **Section 2's branch protection rule**, selecting
   the now-visible required status checks.
5. Replace `@OWNER` in `.github/CODEOWNERS` and the Discussions link in
   `.github/ISSUE_TEMPLATE/config.yml` with real values.
6. (Public repos only) Add the social preview image from **Section 3**.

Remember: `output/*.wav`, `synth_mix.wav`, and `reference_master.wav` are
gitignored (generated artifacts) — run `python make_synthetic_mix.py` after
cloning to regenerate them locally.
