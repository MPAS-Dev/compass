# COMPASS Agent Instructions

These instructions apply to the whole repository unless a deeper
`AGENTS.md` overrides them.

## Source of truth

- Follow the repo's automated style and lint configuration in
  `pyproject.toml`, `.flake8.cfg` and `.pre-commit-config.yaml`.
- If an instruction here conflicts with automated tooling, follow the
  automated tooling.

## Environment

- If `pixi-env/` exists, it is the preferred development environment for
  Python, linting, and `pre-commit`. It is created by `./deploy.py`.
- AI agents should not run `./deploy.py` to create `pixi-env/`
  themselves. Creating or refreshing `pixi-env/` is a developer action.
- Prefer running tools from `pixi-env/.pixi/envs/default/bin/` (for
  example `python`, `pre-commit`, `flake8`, `isort`, and `mypy`) instead
  of relying on the system environment.
- Only fall back to other Python environments if `pixi-env/` does not
  exist or is clearly incomplete.

## Python style

- Keep Python lines at 79 characters or fewer whenever possible.
- Follow `flake8` and `isort` as configured. Do not preserve manual
  formatting or import ordering that `isort` would rewrite.
- Keep imports at module scope whenever possible. Avoid local imports
  unless they are needed to prevent circular imports, defer expensive
  dependencies, or avoid optional dependency failures.
- Avoid nested functions whenever possible. Prefer private module-level
  helpers instead.
- Put public functions before private helper functions whenever
  practical.
- Name private helper functions with a leading underscore when that fits
  existing repo conventions.

## Documentation

- The documentation is reStructuredText built with Sphinx. Follow
  `docs/developers_guide/docs.rst`, which is the authority on structure
  and on the label conventions for cores, configurations and test cases.
- Every new core, configuration or test case needs documentation in both
  the User's Guide and the Developer's Guide, in the same pull request as
  the code.
- Add new or modified classes, methods and functions to the relevant
  `api.rst` (`docs/developers_guide/api.rst` or the core's own, such as
  `docs/developers_guide/ocean/api.rst`).
- Prefer starting from the nearest existing page for a similar test case
  rather than writing a documentation page from scratch.

## GitHub pull requests and issues

- Do not hard-wrap. Write each paragraph and each bullet as a single
  line, however long. GitHub wraps them for display, and hard breaks
  make later edits show up as reflowed paragraphs in the diff.
- Start with a paragraph summarizing what the pull request or issue is
  about, then use sections for the detail.
- Keep the description in a file at the root of the worktree for the
  branch it describes, and never commit it. It is a draft to paste into
  GitHub, not part of the branch's content.
- Follow `.github/pull_request_template.md`: the description goes at
  the top, keep only the checklist lines that apply, and use closing
  keywords for any issue the pull request fixes.
- Do not list individual commits in a pull request description. The
  commits are already on the pull request; describe what the change
  accomplishes as a whole instead.
- Do not describe testing in a pull request description. Testing goes
  in its own `Testing` comment on the pull request, which is what the
  template's checklist asks for.
- An issue should say what happens, what was expected instead, and
  enough about the configuration and commands used to reproduce it.

## Writing for human readers

These rules apply to anything a colleague reads: GitHub comments, pull
request descriptions, issues, plans, design notes. Not code comments or
commit messages, where a reader who wants the mechanism is already in the
right place. Per-artifact rules and worked examples are in
`.claude/skills/<artifact>/SKILL.md`, as plain markdown. Claude Code loads
the matching one automatically; other agents should read it before writing.

Write less; do not pack the same content into denser sentences. Keep
headings, tables and links. Colleagues mostly write unstructured prose, and
structure is an improvement on it. The problem is length.

- **Lead with the answer.** The first two sentences say what you found,
  changed, or propose. Setup and reproduction go last.
- **One point per paragraph, and few paragraphs.** Colleagues write one to
  three per comment; recent AI-written ones ran to eighteen. That gap is
  the complaint. Say each thing once.
- **Do not narrate the mechanism.** The chain of calls, and why the fix is
  right, go in the commit message. Here, say what broke and where to look.
- **Cut clauses that qualify rather than inform**, and any sentence whose
  only job is to justify the one before it. One clause per sentence where
  one will do.
- **Use backticks about half as often as feels natural.** They are for what
  a reader would type or grep. Code blocks hold artifacts you did not
  write, never authored prose.
- **One document, one decision.** Anything still relevant after this merges
  is an issue, not a comment.

Sign anything posted to GitHub on someone's behalf:

```
---

*Posted by <agent> on @<user>'s behalf. The testing, analysis and wording
above are AI-authored; please check them accordingly.*
```

Name the agent, not the vendor: `Claude Code`, `Codex`, and so on.

## Supported machines

- The table under `Supported Machines` in
  `docs/developers_guide/machines/index.rst` is the source of the
  supported machine list in the Developer's Guide. Update it whenever
  machines are added or removed, or compilers and MPI libraries are
  added, removed, or renamed.
- Keep it consistent with the machine config files in
  `compass/machines/`: the `mpi_<compiler>` options under `[deploy]`
  define the valid compiler and MPI combinations, and the
  `<compiler>_<mpi>_target` options under `[build]`, where present,
  define the MPAS make targets.
- Keep `deploy/albany_supported.txt` consistent with the machines and
  compilers that actually support Albany, since MALI functionality
  depends on it.
- When a compiler is added or renamed, update every place it appears:
  the machine config file in `compass/machines/`, the machine pages in
  both the User's and Developer's Guides, and any `load_compass_*.sh`
  examples in the documentation and tutorials.

## Contracts

- Treat `deploy.py` and `deploy/cli_spec.json` as contract files shared
  with the `mache` package.
- Do not modify `deploy.py` or `deploy/cli_spec.json` directly in
  Compass.
- If a change appears necessary, stop and note that the change must be
  made in `mache` first, then synced back into Compass using the normal
  upstream update process.

## Validation

- Run pre-commit on changed files is required before finishing; if sandboxed
  execution fails, request escalation and do not close the task until it has
  run or the user declines.
- Prefer fixing lint and formatting issues rather than suppressing them.
