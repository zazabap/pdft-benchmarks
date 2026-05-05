# Repo Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prune the working tree to the two paper experiments and reorganize so each experiment is self-contained under `results/<experiment>/`, archiving the rest on a `pre-prune-archive` branch.

**Architecture:** Two phases now that PR #7 has squash-merged the writeup work onto main. Spec and plan stay **local-only** (never pushed, never merged):
1. Cut `pre-prune-archive` from current `main` (post-PR #7 squash `1634b4b`) and add a final archive-only commit restoring the three planning notes (`docs/paper-issue-draft.md`, `docs/paper-issue-2-topology.md`, `docs/paper-recommendations.md`) from `local/reorg-docs`. Push.
2. Branch `chore/repo-reorg` from `main` and execute the mechanical reorg commits 1–9. The spec and plan are NOT cherry-picked; they remain only on `local/reorg-docs`.

The original Phase A (rewind writeup branch, recommit writeup-only items, squash-merge) is **superseded by PR #7** — those items are already on main via squash commit `1634b4b`. The original Phase #6 → close + open #7 dance was completed 2026-05-05.

**Tech Stack:** git, bash, GitHub CLI (`gh`), Python (light edits), Typst 0.13 (compile verification).

**Spec:** `docs/superpowers/specs/2026-05-05-repo-reorg-design.md`

**State at plan-revise time (post-PR #7):**
- `main` = `origin/main` = `1634b4b` (squash of "QuickDraw PCA-vs-DCT writeup (rebased onto main) (#7)").
- `local/reorg-docs` = `b76234a`. Local-only safekeeping branch holding the spec + plan + paper-issue drafts + paper-recommendations.md + the original (pre-rebase) snapshot history. Never pushed.
- `local/old-main-pre-pr4` = `31d46aa`. Local-only safekeeping for the 3 stale planning commits that diverged from origin/main before sync.
- PR #6 closed; PR #7 merged.
- No untracked or unstaged files in scope for this plan; main is the source of truth.

**End state:**
- `local/reorg-docs` — local only, unchanged.
- `pre-prune-archive` — pushed to origin; pinned to current `main` + 1 archive-only commit (planning notes).
- `chore/repo-reorg` — pushed to origin; commits 1–9 carry out the reorg. Does NOT contain the spec or the plan.
- `main` — receives the reorg PR after merge. Never sees the spec or the plan.

---

## File Structure (reorg phase, unchanged from spec)

### Files moved (history preserved via `git mv`)
- `experiments/quickdraw.py` → `experiments/quickdraw_pca_vs_block_dct.py`
- `scripts/` → `tools/` (entire directory)
- `tables/published_8q_quickdraw_v2.tex` → `results/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw_v2.tex`
- `docs/global_pca_vs_block_dct.typ` → `results/quickdraw_pca_vs_block_dct/writeup.typ`
- `docs/global_pca_vs_block_dct.pdf` → `results/quickdraw_pca_vs_block_dct/writeup.pdf`
- `docs/figures/*.png` → `results/quickdraw_pca_vs_block_dct/figures/*.png`
- `results/published/quickdraw__<basis>/` (×7) → `results/quickdraw_pca_vs_block_dct/by_basis/<basis>/`
- `results/independent_quickdraw_baselines/` → `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_default/`
- `results/independent_quickdraw_baselines_seed7/` → `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_7/`
- `results/independent_quickdraw_baselines_seed123/` → `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_123/`

### Files deleted on the reorg branch (preserved on `pre-prune-archive`)
- Per spec §B.2, unchanged.

### Files created
- `results/div2k_8q_pca_vs_block_dct/README.md` (placeholder)
- `experiments/div2k_8q_pca_vs_block_dct.py` (placeholder stub)

### Files modified (content edits)
- `tools/render_freq_recon_grid.py` — `--out` default + `results/published/quickdraw__{name}/` path + docstring
- `tools/render_paper_table.py` — `--out` default + `--published-root` default + TODO comment
- `tools/render_pca_basis_visualization.py` — `--out` default + docstring
- `tools/render_ar1_examples.py` — hardcoded `out` Path + docstring
- `tools/independent_quickdraw_baselines.py` — `--out` default
- `experiments/quickdraw_pca_vs_block_dct.py` — docstring + argparse description
- `.gitignore` — full rewrite per spec §D
- `README.md` — full rewrite per spec §F

`tools/validate_manifest.py` is left untouched per spec.

---

## Pre-flight check

- [ ] **Step P1: Confirm starting state**

Run:
```bash
cd /home/claude-user/pdft-benchmarks
git status --short
git checkout main
git status --short
git rev-parse HEAD
git rev-parse origin/main
git rev-parse local/reorg-docs
```

Expected:
- Branch: `main`.
- Local HEAD == Origin HEAD == `1634b4b` (PR #7 squash on main).
- `local/reorg-docs` exists at `b76234a`. Verify with: `git show-ref --verify refs/heads/local/reorg-docs`.
- `git status` clean.
- These four files are tracked on main: `docs/figures/ar1_examples.png`, `docs/global_pca_vs_block_dct.pdf`, `docs/global_pca_vs_block_dct.typ`, `tables/published_8q_quickdraw_v2.tex`, `scripts/render_ar1_examples.py`.

If state differs, stop and reconcile before continuing.

- [ ] **Step P2: Confirm typst is available**

Run: `typst --version`
Expected: `typst 0.13.x` or newer.

- [ ] **Step P3: Confirm GitHub CLI is authenticated**

Run: `gh auth status`
Expected: logged in to github.com with repo scope.

---

## Phase A — DONE (superseded by PR #7 squash, 2026-05-05)

The original Phase A (rewind `docs/quickdraw-pca-dct-writeup`, recommit only writeup-relevant items, squash-merge PR #6) is **complete**. Sequence of events:

1. PR #4 ("Publishable benchmark results: 7 bases × 3 datasets matrix") squash-merged onto `main` → `08d5694`. This absorbed the `bench/publishable-results` stack.
2. PR #6 became unmergeable because its remaining commits stacked on top of bench/publishable-results work that was now content-equivalent to PR #4's squash.
3. The local writeup branch was rebased `--onto origin/main bench/publishable-results docs/quickdraw-pca-dct-writeup`, dropping the duplicate commits and keeping only the writeup-specific delta (`e493922` + `2d1bfad`, replayed as `729fccf` + `93bd635`). One conflict on `tables/published_8q_quickdraw_v2.tex` (modify-vs-delete) resolved by accepting the modified version.
4. Force-push to `docs/quickdraw-pca-dct-writeup` was blocked by the harness; instead the rebased branch was pushed as `docs/quickdraw-pca-dct-writeup-v2`, opened as PR #7, and PR #6 was closed with a comment pointing to #7.
5. PR #7 squash-merged onto `main` → `1634b4b`.

**Net effect on `main`:** writeup typ + figures + render scripts + AR1 figure + writeup PDF + table v2 + independent rerun outputs all landed via the PR #7 squash.

**Net effect on `local/reorg-docs`:** holds the spec + plan + paper-issue drafts + paper-recommendations.md. Never pushed. Source of truth for everything that was deliberately kept off main.

Skip directly to Phase B.

---

## Phase B — Cut and push the archive branch from updated `main`

### Task B1: Confirm we're on a clean main (already done in pre-flight, but re-verify)

- [ ] **Step B1.1: On `main`, fully synced with origin**

Run:
```bash
git checkout main
git fetch origin main
git rev-parse HEAD
git rev-parse origin/main
git status --short
```

Expected: both SHAs match; status clean. SHA is `1634b4b` (PR #7 squash) unless further commits have landed.

- [ ] **Step B1.2: Spot-check the writeup files made it to main**

Run:
```bash
ls docs/global_pca_vs_block_dct.typ docs/figures/ar1_examples.png scripts/render_ar1_examples.py results/independent_quickdraw_baselines/REPORT.md
```

Expected: every file exists.

### Task B2: Cut `pre-prune-archive` and add the planning-notes commit

- [ ] **Step B2.1: Branch from `main`**

Run:
```bash
git checkout -b pre-prune-archive
git rev-parse HEAD
```

Expected: matches `main`'s SHA from B1.1.

- [ ] **Step B2.2: Restore the three planning-only files from `local/reorg-docs`**

These three never made it onto main (intentionally, per the user's "docs only updated locally" preference). They live only on `local/reorg-docs` and need to be on the archive so the spec's "preserved on archive" intent is honored.

Run:
```bash
git checkout local/reorg-docs -- \
    docs/paper-issue-draft.md \
    docs/paper-issue-2-topology.md \
    docs/paper-recommendations.md
ls docs/paper-issue-draft.md docs/paper-issue-2-topology.md docs/paper-recommendations.md
git status --short
```

Expected: all three files exist; all three staged as new (`A`).

- [ ] **Step B2.3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
archive: paper-issue planning notes + paper-recommendations.md

These three markdown files were planning notes that never made it to
main: paper-issue-draft.md and paper-issue-2-topology.md were
intentionally kept local during PR #6/#7 rebase; paper-recommendations.md
was on the writeup branch but pruned during the rebase onto main.

Per the repo-reorg spec
(docs/superpowers/specs/2026-05-05-repo-reorg-design.md §B.2), they
are preserved on this archive branch but deliberately not merged to
main. Recover with:

    git checkout pre-prune-archive -- docs/paper-issue-draft.md \
        docs/paper-issue-2-topology.md docs/paper-recommendations.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B2.4: Push**

Run:
```bash
git push -u origin pre-prune-archive
git ls-remote origin pre-prune-archive
```

Expected: branch on origin; SHA matches the local archive commit. The archive is now durable; deletions in Phase D are safe.

Note: this archive does NOT contain `docs/superpowers/specs/2026-05-05-repo-reorg-design.md` or `docs/superpowers/plans/2026-05-05-repo-reorg.md`. Per the user's preference, those documents stay only on `local/reorg-docs` (never pushed). If you ever need them back, `git checkout local/reorg-docs -- <path>`.

---

## Phase C — Cut the reorg branch from `main`

### Task C1: Branch `chore/repo-reorg` from `main`

- [ ] **Step C1.1: Switch back to `main`**

Run:
```bash
git checkout main
git rev-parse HEAD
```

Expected: matches B1.1's SHA (still the post-squash `main`).

- [ ] **Step C1.2: Branch and switch**

Run:
```bash
git checkout -b chore/repo-reorg
git branch --show-current
```

Expected: `chore/repo-reorg`.

- [ ] **Step C1.3: Confirm the spec, plan, paper-issue drafts, and paper-recommendations.md are NOT in this branch's tree**

Run:
```bash
ls docs/superpowers/specs/2026-05-05-repo-reorg-design.md 2>&1
ls docs/superpowers/plans/2026-05-05-repo-reorg.md 2>&1
ls docs/paper-issue-draft.md docs/paper-issue-2-topology.md docs/paper-recommendations.md 2>&1
```

Expected: every entry reports `No such file or directory`. Spec + plan stay on `local/reorg-docs`; paper-issue drafts and paper-recommendations.md stay on `pre-prune-archive`.

If any of them are present here, something went wrong upstream — investigate before continuing.

The reorg branch executes purely on tree changes (renames, moves, deletes, content edits) without carrying the planning material.

---

## Phase D — Reorg execution (Tasks D1–D9)

The branch is `chore/repo-reorg`, branched from `main` at `1634b4b` (post-PR #7 squash). No spec/plan content on this branch.

### Task D1: Rename `scripts/` → `tools/`

- [ ] **Step D1.1: Run the directory rename**

Run:
```bash
git mv scripts tools
git status --short | head -20
ls tools/
```

Expected: `git status` shows `R  scripts/<file> -> tools/<file>` for each tracked file. `ls tools/` lists every script that was tracked under `scripts/`.

- [ ] **Step D1.2: Clean stray `scripts/` if a `__pycache__` survived**

Run: `ls scripts/ 2>&1`
Expected: directory does not exist, OR contains only `__pycache__`. If the latter: `rm -rf scripts/`.

- [ ] **Step D1.3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: rename scripts/ → tools/

Per the repo-reorg spec: 'scripts/' becomes 'tools/' to better signal
that these are CLI executables, not library code. Internal path
references inside these files are updated in a later commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step D1.4: Verify history preserved**

Run: `git log --follow --oneline tools/render_paper_table.py | head -3`
Expected: ≥2 commits including pre-rename history.

### Task D2: Rename `experiments/quickdraw.py` and refocus its docstring

- [ ] **Step D2.1: Rename**

Run:
```bash
git mv experiments/quickdraw.py experiments/quickdraw_pca_vs_block_dct.py
```

- [ ] **Step D2.2: Replace the docstring**

Edit `experiments/quickdraw_pca_vs_block_dct.py`. Replace the existing module docstring (the one starting `"""QuickDraw benchmark...` and ending `...m=5 thanks to the asymmetric...image."""`) with:

```python
"""QuickDraw PCA-vs-block-DCT benchmark (m=n=5, 32×32).

Trains the seven registered parametric bases (qft, entangled_qft, tebd,
mera, blocked, rich, real_rich) and evaluates each against the four
classical baselines: global FFT, global DCT, 8×8-block FFT, 8×8-block
DCT. PCA + block-DCT are the comparison anchors for the paper; the
parametric bases are the candidates being assessed against them.

Outputs land in `results/quickdraw_pca_vs_block_dct/` when run with the
default --out (None → uses the run_experiment default which writes
under results/<dataset>_<preset>_<timestamp>/; pass --out explicitly
to drop straight into the canonical paper directory).

`mera` is silently skipped by run_experiment because m+n=10 is not a
power of 2. The block bases (blocked, rich, real_rich) train at m=5
thanks to the asymmetric `_blocked` split (inner_m=3, block_log_m=2)
in pdft_benchmarks.bases — a 4×4 grid of 8×8 blocks fitting a 32×32
image.
"""
```

- [ ] **Step D2.3: Verify the file still parses**

Run:
```bash
python -c "import ast, pathlib; ast.parse(pathlib.Path('experiments/quickdraw_pca_vs_block_dct.py').read_text())" && echo "parse OK"
```

Expected: `parse OK`.

- [ ] **Step D2.4: Commit**

Run:
```bash
git add experiments/quickdraw_pca_vs_block_dct.py
git commit -m "$(cat <<'EOF'
chore: rename quickdraw.py → quickdraw_pca_vs_block_dct.py + reframe docstring

Per the repo-reorg spec. The actual experiment (7 bases vs. 4 classical
baselines) is unchanged; only the file name and its docstring are
updated to reflect the paper's PCA-vs-block-DCT framing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D3: Move `results/published/quickdraw__*` into `by_basis/`

- [ ] **Step D3.1: Run the seven moves**

Run:
```bash
git mv results/published/quickdraw__qft results/quickdraw_pca_vs_block_dct/by_basis/qft
git mv results/published/quickdraw__entangled_qft results/quickdraw_pca_vs_block_dct/by_basis/entangled_qft
git mv results/published/quickdraw__tebd results/quickdraw_pca_vs_block_dct/by_basis/tebd
git mv results/published/quickdraw__mera results/quickdraw_pca_vs_block_dct/by_basis/mera
git mv results/published/quickdraw__blocked results/quickdraw_pca_vs_block_dct/by_basis/blocked
git mv results/published/quickdraw__rich results/quickdraw_pca_vs_block_dct/by_basis/rich
git mv results/published/quickdraw__real_rich results/quickdraw_pca_vs_block_dct/by_basis/real_rich
```

`git mv` will create the parent `by_basis/` automatically.

- [ ] **Step D3.2: Verify**

Run:
```bash
ls results/quickdraw_pca_vs_block_dct/by_basis/
git status --short | head -20
```

Expected: 7 subdirs in `by_basis/`.

- [ ] **Step D3.3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: move results/published/quickdraw__* into by_basis/

All 7 cells (qft, entangled_qft, tebd, mera, blocked, rich, real_rich)
move under results/quickdraw_pca_vs_block_dct/by_basis/<basis>/. They
are the comparison rows of the paper writeup table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D4: Move `results/independent_quickdraw_baselines*` into `independent_reruns/`

- [ ] **Step D4.1: Run the three moves**

Run:
```bash
git mv results/independent_quickdraw_baselines results/quickdraw_pca_vs_block_dct/independent_reruns/seed_default
git mv results/independent_quickdraw_baselines_seed7 results/quickdraw_pca_vs_block_dct/independent_reruns/seed_7
git mv results/independent_quickdraw_baselines_seed123 results/quickdraw_pca_vs_block_dct/independent_reruns/seed_123
```

- [ ] **Step D4.2: Verify**

Run:
```bash
ls results/quickdraw_pca_vs_block_dct/independent_reruns/
```

Expected: `seed_7`, `seed_123`, `seed_default`.

- [ ] **Step D4.3: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: move independent_quickdraw_baselines* into independent_reruns/

Three independent rerun directories (default, seed=7, seed=123) move
under results/quickdraw_pca_vs_block_dct/independent_reruns/seed_<n>/
to live alongside the experiment they verify.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D5: Move tables, figures, writeup; verify `typst compile`

- [ ] **Step D5.1: Move the LaTeX table**

Run:
```bash
mkdir -p results/quickdraw_pca_vs_block_dct/tables
git mv tables/published_8q_quickdraw_v2.tex results/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw_v2.tex
```

- [ ] **Step D5.2: Move the figures**

Run:
```bash
mkdir -p results/quickdraw_pca_vs_block_dct/figures
git mv docs/figures/ar1_examples.png            results/quickdraw_pca_vs_block_dct/figures/
git mv docs/figures/freq_recon_grid_img0.png    results/quickdraw_pca_vs_block_dct/figures/
git mv docs/figures/freq_recon_grid_img0_freq.png results/quickdraw_pca_vs_block_dct/figures/
git mv docs/figures/freq_recon_grid_img2.png    results/quickdraw_pca_vs_block_dct/figures/
git mv docs/figures/freq_recon_grid_img2_freq.png results/quickdraw_pca_vs_block_dct/figures/
git mv docs/figures/pca_basis.png               results/quickdraw_pca_vs_block_dct/figures/
ls docs/figures/ 2>&1
```

Expected: last `ls` shows the directory empty or absent.

- [ ] **Step D5.3: Move the writeup typ + pdf**

Run:
```bash
git mv docs/global_pca_vs_block_dct.typ results/quickdraw_pca_vs_block_dct/writeup.typ
git mv docs/global_pca_vs_block_dct.pdf results/quickdraw_pca_vs_block_dct/writeup.pdf
```

- [ ] **Step D5.4: Verify image references resolve**

Run:
```bash
grep -n 'image("figures/' results/quickdraw_pca_vs_block_dct/writeup.typ
ls results/quickdraw_pca_vs_block_dct/figures/
```

Expected: every `image("figures/<name>")` reference has a matching file in `figures/`.

- [ ] **Step D5.5: Verify typst compile succeeds**

Run:
```bash
cd results/quickdraw_pca_vs_block_dct
typst compile writeup.typ writeup.pdf
cd /home/claude-user/pdft-benchmarks
```

Expected: typst exits 0; PDF regenerated.

- [ ] **Step D5.6: Stage the regenerated PDF if it changed**

Run: `git status --short`
If `writeup.pdf` shows `M`, run `git add results/quickdraw_pca_vs_block_dct/writeup.pdf`.

- [ ] **Step D5.7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: move tables, figures, writeup into results/quickdraw_pca_vs_block_dct/

- tables/published_8q_quickdraw_v2.tex → tables/
- docs/figures/*.png (6) → figures/
- docs/global_pca_vs_block_dct.{typ,pdf} → writeup.{typ,pdf}

Image refs in writeup.typ are relative ('figures/...') so they
resolve unchanged after the move. typst compile verified.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D6: Prune deleted paths

The archive branch is on origin (Phase B), so deletions are safe.

- [ ] **Step D6.1: Remove non-paper experiments**

Run:
```bash
git rm experiments/div2k_10q_block.py
git rm experiments/div2k_10q_circuit.py
git rm experiments/div2k_8q_blocked_vs_blockdct.py
git rm experiments/div2k_8q_circuit_vs_classical.py
git rm experiments/post_run_analysis.py
```

- [ ] **Step D6.2: Remove obsolete tools (post-rename paths)**

Run:
```bash
git rm tools/cpu_vs_gpu_batched.py
git rm tools/extract_canonical_cells.py
git rm tools/render_published_readme.py
git rm tools/run_canonical.sh
git rm tools/sweep_qft_config.py
```

(`tools/append_pca_to_archived_runs.py` and `tools/verify_dct_klt_convergence.py` were on the writeup branch but are NOT on main — no `git rm` needed.)

- [ ] **Step D6.3: Remove old table version**

Run:
```bash
git rm tables/published_8q_quickdraw.tex
ls tables/ 2>&1
```

Expected: directory does not exist.

- [ ] **Step D6.4: Remove `diagrams/`**

Run: `git rm -r diagrams/`

- [ ] **Step D6.5: Remove non-paper published cells**

Run:
```bash
for d in div2k_8q__blocked div2k_8q__entangled_qft div2k_8q__mera div2k_8q__qft \
         div2k_8q__real_rich div2k_8q__rich div2k_8q__tebd \
         div2k_10q__blocked div2k_10q__entangled_qft div2k_10q__mera \
         div2k_10q__qft div2k_10q__real_rich div2k_10q__rich div2k_10q__tebd; do
    git rm -r results/published/$d
done
git rm results/published/MANIFEST.json
git rm results/published/README.md
ls results/published/ 2>&1
```

Expected: directory does not exist.

- [ ] **Step D6.6: Remove other non-paper results**

Run:
```bash
git rm -r results/_archive
git rm -r results/ablations
git rm -r results/div2k_10q_generalized_20260427-061939
ls results/ 2>&1
```

Expected: only `quickdraw_pca_vs_block_dct/` remains.

- [ ] **Step D6.7: Verify `docs/figures/` is gone (no top-level docs/*.md to remove)**

Run:
```bash
ls docs/figures/ 2>&1
ls docs/paper-recommendations.md docs/paper-issue-draft.md docs/paper-issue-2-topology.md 2>&1
```

Expected: `docs/figures/` does not exist (was emptied in D5.2). The three planning markdown files do not exist either — they were never on main; they live only on `pre-prune-archive`. No `git rm` needed for any of them.

- [ ] **Step D6.8: Clean stray `__pycache__` directories on disk**

Run:
```bash
rm -rf __pycache__/ experiments/__pycache__/ tools/__pycache__/ \
       src/pdft_benchmarks/__pycache__/ src/pdft_benchmarks/datasets/__pycache__/ \
       tests/__pycache__/ tests/datasets/__pycache__/ 2>/dev/null
git status --short | grep -i pycache
```

Expected: no pycache entries.

- [ ] **Step D6.9: Verify the working-tree shape**

Run:
```bash
ls /home/claude-user/pdft-benchmarks
ls results/
ls experiments/
ls tools/ | wc -l
```

Expected:
- Top level: `LICENSE`, `README.md`, `docs`, `experiments`, `pyproject.toml`, `results`, `src`, `tests`, `tools`.
- `results/`: only `quickdraw_pca_vs_block_dct`.
- `experiments/`: only `quickdraw_pca_vs_block_dct.py`.
- `tools/`: 6 entries (`render_paper_table.py`, `render_freq_recon_grid.py`, `render_pca_basis_visualization.py`, `render_ar1_examples.py`, `independent_quickdraw_baselines.py`, `validate_manifest.py`).

- [ ] **Step D6.10: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore: prune non-paper code and results

Removes from working tree (preserved on pre-prune-archive):
- non-paper experiments (div2k 8q/10q variants, post_run_analysis)
- obsolete tools (canonical-tree extractors, sweeps, perf one-offs)
- v1 paper table superseded by v2
- diagrams/, tables/, results/_archive, results/ablations
- non-paper published cells (div2k_8q__*, div2k_10q__*) and the
  pub manifest/README that indexed the now-dismantled tree

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D7: Update internal path references

`tools/validate_manifest.py` is left untouched per spec.
`experiments/quickdraw_pca_vs_block_dct.py` already has its docstring updated; no path edits needed (it uses the library's `output_dir=None` default).

- [ ] **Step D7.1: Edit `tools/render_freq_recon_grid.py` (`--out` default)**

Replace `ap.add_argument("--out", default="docs/figures/freq_recon_grid.png")` with `ap.add_argument("--out", default="results/quickdraw_pca_vs_block_dct/figures/freq_recon_grid.png")`.

- [ ] **Step D7.2: Edit `tools/render_freq_recon_grid.py` (trained-bases path + docstring)**

Replace `path = Path(f"results/published/quickdraw__{name}/trained_{name}.json")` with `path = Path(f"results/quickdraw_pca_vs_block_dct/by_basis/{name}/trained_{name}.json")`.

Replace `Loads trained bases from results/published/quickdraw__{name}/trained_{name}.json,` with `Loads trained bases from results/quickdraw_pca_vs_block_dct/by_basis/{name}/trained_{name}.json,`.

- [ ] **Step D7.3: Edit `tools/render_paper_table.py`**

Replace `parser.add_argument("--published-root", default="results/published", type=Path)` with `parser.add_argument("--published-root", default="results/quickdraw_pca_vs_block_dct/by_basis", type=Path)`.

Replace `parser.add_argument("--out", default="tables/published_8q_quickdraw.tex", type=Path)` with `parser.add_argument("--out", default="results/quickdraw_pca_vs_block_dct/tables/published_8q_quickdraw.tex", type=Path)`.

Add a TODO comment immediately above the `--published-root` argparse line:

```python
    # TODO(repo-reorg): walks <root>/<dataset>__<basis>/; new layout is
    # <root>/<basis>/. Rework before next paper-table regen.
```

- [ ] **Step D7.4: Edit `tools/render_pca_basis_visualization.py`**

Replace `ap.add_argument("--out", default="docs/figures/pca_basis.png")` with `ap.add_argument("--out", default="results/quickdraw_pca_vs_block_dct/figures/pca_basis.png")`.

Replace `  docs/figures/pca_basis.png — 3-panel figure showing` with `  results/quickdraw_pca_vs_block_dct/figures/pca_basis.png — 3-panel figure showing`.

- [ ] **Step D7.5: Edit `tools/render_ar1_examples.py`**

Replace `    out = Path("docs/figures/ar1_examples.png")` with `    out = Path("results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png")`.

Replace `Output: docs/figures/ar1_examples.png — three 64×64 patches sampled` with `Output: results/quickdraw_pca_vs_block_dct/figures/ar1_examples.png — three 64×64 patches sampled`.

- [ ] **Step D7.6: Edit `tools/independent_quickdraw_baselines.py`**

Replace `    ap.add_argument("--out", default="results/independent_quickdraw_baselines",` with `    ap.add_argument("--out", default="results/quickdraw_pca_vs_block_dct/independent_reruns/seed_default",`.

- [ ] **Step D7.7: Sanity-check no remaining files reference dead paths**

Run:
```bash
grep -rn "docs/figures\|results/published\|results/independent_quickdraw\|tables/published_8q_quickdraw" tools/ experiments/ 2>/dev/null | grep -v __pycache__ | grep -v validate_manifest.py
```

Expected: empty output (only `validate_manifest.py` retains a `results/published` reference, intentional per spec).

- [ ] **Step D7.8: Smoke-import each edited file**

Run:
```bash
for f in tools/render_freq_recon_grid.py tools/render_paper_table.py \
         tools/render_pca_basis_visualization.py tools/render_ar1_examples.py \
         tools/independent_quickdraw_baselines.py; do
    python -c "import ast, pathlib; ast.parse(pathlib.Path('$f').read_text())" && echo "$f parse OK"
done
```

Expected: each prints `<file> parse OK`.

- [ ] **Step D7.9: Commit**

Run:
```bash
git add tools/render_freq_recon_grid.py tools/render_paper_table.py \
        tools/render_pca_basis_visualization.py tools/render_ar1_examples.py \
        tools/independent_quickdraw_baselines.py
git commit -m "$(cat <<'EOF'
chore: update tool path defaults to new results layout

Re-points hardcoded output paths from docs/figures/, tables/, and
results/published/quickdraw__<basis>/ to the new colocated layout
under results/quickdraw_pca_vs_block_dct/.

render_paper_table.py: defaults updated, but its directory-walk logic
still expects the old <root>/<dataset>__<basis>/ shape. TODO comment
added; full rework deferred to a follow-on.

validate_manifest.py left as-is per the reorg spec; its target
manifest no longer exists, but the user wanted the file kept.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D8: Rewrite `.gitignore`

- [ ] **Step D8.1: Replace `.gitignore`**

Replace the entire contents of `.gitignore` with:

```gitignore
# Build / cache
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.cache/
.jax_cache/

# Local data dumps
*.npz
*.json.gz

# results/ — track curated text artifacts (metrics, csvs, figures, tables,
# writeup); ignore heavy regeneratable blobs.
results/**/trained_*.json
results/**/loss_history/
results/**/run.log
results/**/failures/
```

- [ ] **Step D8.2: Verify nothing tracked becomes ignored**

Run:
```bash
git ls-files --error-unmatch results/quickdraw_pca_vs_block_dct/by_basis/qft/metrics.json
git ls-files --error-unmatch results/quickdraw_pca_vs_block_dct/figures/pca_basis.png
git status --short
```

Expected: both `ls-files` checks succeed; `git status` shows only `.gitignore` modified.

- [ ] **Step D8.3: Commit**

Run:
```bash
git add .gitignore
git commit -m "$(cat <<'EOF'
chore: rewrite .gitignore for new layout

Drops the elaborate results/published/, results/ablations/, docs/figures/
allowlist rules — those paths no longer exist. New rules track curated
text artifacts under results/<exp>/ and ignore heavy regeneratable
blobs (trained_*.json, loss_history/, run.log, failures/).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task D9: Rewrite README, add DIV2K-8q placeholders

- [ ] **Step D9.1: Replace `README.md`** with the content per spec §F (full content reproduced inline below — copy verbatim):

```markdown
# pdft Benchmarks — PCA vs block-DCT

Benchmark code and results backing the paper-section comparison of the
parametric quantum-circuit basis family against PCA and 8×8 block-DCT
classical baselines.

## Experiments

1. **QuickDraw** (m=n=5, 32×32) — implemented. Results, figures,
   table, and writeup live in `results/quickdraw_pca_vs_block_dct/`.
2. **DIV2K-8q** (m=n=8, 256×256) — placeholder. The paper needs a
   DIV2K-8q analog of the QuickDraw experiment, including MERA on the
   unblocked variant. See `results/div2k_8q_pca_vs_block_dct/README.md`
   and the follow-on spec.

## Layout

```
experiments/         Runnable entry points (one per paper experiment).
src/pdft_benchmarks/ Library: bases, baselines, pipeline, evaluation, PCA.
tools/               CLI utilities: renderers, validators, independent reruns.
results/<exp>/       Self-contained per-experiment outputs:
                       metrics.json, *.csv, figures/, tables/, writeup.{typ,pdf},
                       by_basis/<basis>/, independent_reruns/seed_*/
docs/superpowers/    Specs and implementation plans.
tests/               Unit + integration tests.
```

## Running

\`\`\`bash
# Train and evaluate the QuickDraw experiment
python experiments/quickdraw_pca_vs_block_dct.py --gpu 0 \
    --out results/quickdraw_pca_vs_block_dct

# Re-render the paper figures from existing trained bases
python tools/render_freq_recon_grid.py
python tools/render_pca_basis_visualization.py
python tools/render_ar1_examples.py

# Re-compile the writeup
typst compile results/quickdraw_pca_vs_block_dct/writeup.typ

# Independent rerun for verification (≈5 s/seed)
python tools/independent_quickdraw_baselines.py --seed 42
\`\`\`

## Install

\`\`\`bash
pip install -e ".[bench,gpu]"   # GPU
pip install -e ".[bench]"        # CPU-only smoke
\`\`\`

## Tests

\`\`\`bash
pytest tests/ --no-cov                   # Layer A: <30 s, no GPU, no datasets
pytest tests/ -m integration --no-cov    # Layer B: requires datasets, optional GPU
\`\`\`

## Archive

Pre-reorg state (full canonical 7-basis × 3-dataset matrix, ablations,
DIV2K-10q runs, paper-issue planning notes) is preserved on the
`pre-prune-archive` branch on origin. Recover anything from there with
`git checkout pre-prune-archive -- <path>`.

## Datasets

The harness reads from `/home/claude-user/ParametricDFT-Benchmarks.jl/data/`:

- `quickdraw/*.npy` — 5 categories of 28×28 uint8 drawings.
- `DIV2K_train_HR/*.png` — high-resolution PNGs (cropped + resized to 256×256).

Adjust the `data_root=` defaults in the loader functions to use a
different path.
```

(Note: when copying into the actual file, replace the escaped `\`\`\`bash` fences with real triple-backtick fences. They are escaped here only so this plan's markdown renders cleanly.)

- [ ] **Step D9.2: Create `results/div2k_8q_pca_vs_block_dct/README.md`**

```markdown
# DIV2K-8q PCA-vs-block-DCT (placeholder)

This directory will hold the DIV2K-8q analog of the QuickDraw
PCA-vs-block-DCT experiment, including MERA on the unblocked variant.

The follow-on spec defines the experiment template (matching
`results/quickdraw_pca_vs_block_dct/`) and the runnable
`experiments/div2k_8q_pca_vs_block_dct.py`. Until that work lands,
this directory is intentionally empty apart from this README.
```

- [ ] **Step D9.3: Create `experiments/div2k_8q_pca_vs_block_dct.py`**

```python
#!/usr/bin/env python3
"""DIV2K-8q PCA-vs-block-DCT benchmark — placeholder.

The paper requires a DIV2K-8q (m=n=8, 256×256) analog of the QuickDraw
experiment, including MERA on the unblocked variant. The implementation
is deferred to a follow-on spec; this stub exists so the file path is
reserved and importers fail loudly rather than silently.
"""

import sys


def main() -> int:
    print(
        "experiments/div2k_8q_pca_vs_block_dct.py is a placeholder. "
        "The DIV2K-8q PCA-vs-block-DCT experiment is deferred to a "
        "follow-on spec; see results/div2k_8q_pca_vs_block_dct/README.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step D9.4: Verify the placeholder script runs and exits non-zero**

Run:
```bash
python experiments/div2k_8q_pca_vs_block_dct.py; echo "exit=$?"
```

Expected: prints the placeholder message; `exit=2`.

- [ ] **Step D9.5: Verify final tree shape**

Run:
```bash
ls /home/claude-user/pdft-benchmarks
ls experiments/
ls results/
ls tools/
```

Expected:
- Top level: `LICENSE`, `README.md`, `docs`, `experiments`, `pyproject.toml`, `results`, `src`, `tests`, `tools`.
- `experiments/`: `div2k_8q_pca_vs_block_dct.py`, `quickdraw_pca_vs_block_dct.py`.
- `results/`: `div2k_8q_pca_vs_block_dct`, `quickdraw_pca_vs_block_dct`.
- `tools/`: 6 files (`render_paper_table.py`, `render_freq_recon_grid.py`, `render_pca_basis_visualization.py`, `render_ar1_examples.py`, `independent_quickdraw_baselines.py`, `validate_manifest.py`).

- [ ] **Step D9.6: Commit**

Run:
```bash
git add README.md results/div2k_8q_pca_vs_block_dct/README.md \
        experiments/div2k_8q_pca_vs_block_dct.py
git commit -m "$(cat <<'EOF'
docs: rewrite README + add DIV2K-8q placeholders

README documents the new layout, the two paper experiments, run/install
instructions, and where to recover pre-reorg state (pre-prune-archive
branch on origin).

DIV2K-8q placeholders reserve experiments/div2k_8q_pca_vs_block_dct.py
and results/div2k_8q_pca_vs_block_dct/README.md for the follow-on
spec (matches QuickDraw template + adds MERA on unblocked variant).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase E — Push and open the reorg PR

- [ ] **Step E1: Push**

Run:
```bash
git push -u origin chore/repo-reorg
```

- [ ] **Step E2: Verify both branches on remote**

Run:
```bash
git ls-remote --heads origin pre-prune-archive chore/repo-reorg
```

Expected: both refs listed.

- [ ] **Step E3: Open the PR**

Run:
```bash
gh pr create --base main --title "Repo reorg: paper-focused prune" --body "$(cat <<'EOF'
## Summary
- Reduces working tree to the QuickDraw paper experiment + a DIV2K-8q placeholder.
- Colocates each experiment's metrics, figures, tables, and writeup under `results/<experiment>/`.
- Renames `scripts/` → `tools/`.
- Pre-reorg state preserved on `pre-prune-archive` (pushed to origin).

Per spec `docs/superpowers/specs/2026-05-05-repo-reorg-design.md` and plan `docs/superpowers/plans/2026-05-05-repo-reorg.md`.

## Test plan
- [ ] `typst compile results/quickdraw_pca_vs_block_dct/writeup.typ` produces an unchanged PDF.
- [ ] `python tools/render_freq_recon_grid.py` writes into `results/quickdraw_pca_vs_block_dct/figures/`.
- [ ] `python tools/independent_quickdraw_baselines.py --seed 42` writes into `results/quickdraw_pca_vs_block_dct/independent_reruns/seed_default/`.
- [ ] `pytest tests/ --no-cov` passes.
- [ ] `pre-prune-archive` exists on origin and contains the deleted material.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Goal — Phase D Tasks D3–D5 (move) + Task D6 (prune) + Task D9 (DIV2K-8q placeholder).
- Final layout (spec §"Final layout") — every entry has a corresponding task in Phase D.
- File-by-file disposition (spec §B.1, §B.2) — every row mapped to a step (D3–D6 for moves and deletes; D1 for the directory rename; D9 for placeholders).
- Branch and git mechanics (spec §C) — Phase A handles the "stage in-flight items first"; Phase B handles archive creation + push; Phase C handles working-branch creation + spec/plan replay; Phase D handles commits 1–9; Phase E handles PR.
- `.gitignore` rewrite (spec §D) — Task D8.
- Internal path updates (spec §E) — Task D7, with `validate_manifest.py` excluded per spec.
- README rewrite (spec §F outline) — Task D9.
- Risks (spec §G) — typst compile in D5.5; archive-before-delete enforced by Phase B preceding D6; `git mv` history preservation verified in D1.4.

**Sequencing under PR #6:** Phase A keeps PR #6 focused on the writeup — spec+plan and the snapshot commit are moved off the writeup branch to `local/reorg-docs`, then only the 5 writeup-relevant items are recommitted. Phase B's archive captures the post-squash state plus paper-issue drafts as a final commit, so the deletes in D6 are reversible. Phase C branches the reorg from `main` without carrying the spec/plan — those documents stay local-only per the user's preference. The reorg PR is purely mechanical tree changes.

**Type / signature consistency:**
- Output paths in D7 edits match the moves in D3–D5.
- Branch name `chore/repo-reorg` consistent across Phase C, Phase D, Phase E.
- Branch name `pre-prune-archive` consistent across Phase B and Step D9 README, Step E3 PR body.

**Placeholder scan:**
- No "TBD"/"TODO" in plan steps. The `# TODO(repo-reorg):` comment in `render_paper_table.py` (Step D7.3) is intentional code-side content, not a missing plan step.
- Every code edit shows `old_string` → `new_string`.
- Every command has an expected outcome.
