# Contributing to DataChef

Thank you for helping improve DataChef. Contributions of all sizes are welcome, including bug fixes, analytical tools, tests, documentation, accessibility improvements, model evaluations, and architectural proposals.

DataChef follows an **issue-first contribution workflow**:

1. Open an issue.
2. Agree on the scope and approach.
3. Implement the change on a dedicated branch.
4. Open a pull request linked to the issue.

Please do not begin substantial implementation work or open an unlinked pull request. Starting with an issue helps prevent duplicate work, gives maintainers an opportunity to clarify requirements, and keeps design decisions visible to the community.

## 1. Open an issue first

Before changing code, open a GitHub issue describing the problem or proposal.

Search existing issues first. If a relevant issue already exists, comment there instead of creating a duplicate.

A useful issue should include:

- A clear title.
- The problem, limitation, or user need.
- Why the change would benefit DataChef.
- Relevant examples, screenshots, logs, or reproduction steps.
- A proposed approach, if you have one.
- Alternatives or trade-offs that should be considered.

For bugs, include:

- Steps to reproduce the problem.
- Expected behavior.
- Actual behavior.
- Operating system and Python version.
- DataChef version or Git commit.
- Inference runtime and model, when relevant.
- A minimal, non-sensitive sample dataset when the bug depends on data.

Never attach API keys, credentials, private datasets, personal information, or other sensitive material to an issue.

## 2. Wait for scope confirmation

Allow a maintainer to review the issue before starting substantial work. The maintainer may:

- Confirm that the change fits the project roadmap.
- Ask for more information.
- Suggest a smaller or different approach.
- Identify related work or dependencies.
- Assign the issue to you or confirm that you may proceed.

An issue being open does not automatically mean its proposed implementation has been accepted. Begin implementation after the scope has been confirmed in the issue.

If the agreed scope changes while you are working, update the issue before expanding the pull request.

## 3. Set up the development environment

Fork and clone the repository, then create an isolated Python environment:

```bash
git clone https://github.com/<your-username>/DataChef.git
cd DataChef
python -m venv .venv
```

Activate the environment and install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` when local inference configuration is required. Never commit `.env` files, tokens, credentials, downloaded model weights, or private datasets.

## 4. Create a dedicated branch

Create one branch per issue. Start from an up-to-date `main` branch.

Recommended branch names:

```text
feature/<issue-number>-short-description
fix/<issue-number>-short-description
docs/<issue-number>-short-description
test/<issue-number>-short-description
refactor/<issue-number>-short-description
```

Examples:

```text
feature/42-add-cramers-v
fix/87-filter-failed-evidence
docs/103-local-model-setup
```

Keep the branch focused on the linked issue. Unrelated cleanup or refactoring should be proposed separately.

## 5. Implement the agreed change

While implementing:

- Follow the scope agreed in the issue.
- Preserve DataChef's architectural boundaries.
- Keep computation deterministic and local to the analytical engine.
- Do not introduce unrestricted code execution.
- Do not send raw dataset rows to an inference provider.
- Validate model-generated tool arguments before execution.
- Add or update tests for changed behavior.
- Update documentation when behavior, configuration, or public APIs change.
- Avoid drive-by formatting or unrelated file changes.

New analytical tools should include:

- A deterministic implementation under `datachef/analytics/`.
- A typed and bounded tool schema.
- Validation rules for arguments and column names.
- Unit tests using independently verified expected results.
- Clear statistical assumptions and limitations.
- Documentation describing when the tool should and should not be used.

## 6. Validate the change locally

Run the relevant tests while developing. Before opening a pull request, run the complete available checks:

```bash
python -m pytest
python -m ruff check datachef tests
python -m mypy datachef
python -m compileall -q datachef
python -m pip check
```

If a check cannot be run or an existing failure is unrelated to your change, explain that clearly in the pull request. Do not hide, delete, or weaken a failing test merely to make the suite pass.

## 7. Commit clearly

Write focused commits that explain the change. Prefer several understandable commits over one large, unrelated commit.

Example commit messages:

```text
fix: exclude failed tool executions from finding evidence
feat: add Cramer's V to categorical association results
docs: document llama.cpp local inference setup
```

Avoid committing generated caches, local investigation records, model weights, secrets, editor files, or private data.

## 8. Open a linked pull request

When the implementation is ready, open a pull request against `main`.

The pull request must link the issue using a closing keyword in its description:

```text
Closes #42
```

Use `Refs #42` instead when the pull request is only part of a larger issue and should not close it.

The pull request description should include:

- What changed.
- Why it changed.
- The linked issue.
- Important design decisions and trade-offs.
- Tests that were added or updated.
- Commands used to verify the change.
- Screenshots or recordings for user-interface changes.
- Performance or model-evaluation results when relevant.
- Known limitations or follow-up work.

Keep pull requests small enough to review. If a change is large, agree on a sequence of smaller pull requests in the issue before implementation.

## Pull request checklist

- [ ] I opened or joined an issue before starting this work.
- [ ] A maintainer confirmed the scope and approach.
- [ ] This pull request links the issue.
- [ ] The change stays within the agreed scope.
- [ ] I added or updated tests for changed behavior.
- [ ] I updated relevant documentation.
- [ ] I ran the applicable test, lint, type, and package checks.
- [ ] I did not commit secrets, private data, model weights, or local artifacts.
- [ ] I documented user-visible, statistical, privacy, or compatibility implications.
- [ ] The pull request contains no unrelated changes.

## 9. Participate in review

Pull requests require review before merging. Reviewers may request changes to correctness, tests, architecture, documentation, privacy, statistical validity, performance, or maintainability.

When responding to review:

- Reply to questions and requested changes.
- Push updates to the same branch and pull request.
- Explain when you disagree and provide technical evidence.
- Resolve conversations only after the concern has been addressed.
- Avoid force-pushing after review unless necessary; if you do, notify reviewers.

Approval does not guarantee immediate merging. Maintainers may wait for required checks, additional reviews, release timing, or dependent work.

## 10. After the pull request is merged

Once the pull request is merged:

- Confirm that the linked issue was closed when appropriate.
- Delete the completed feature branch.
- Open a new issue for any follow-up work that was intentionally left out of scope.
- Do not reuse the merged branch for unrelated changes.

## Core project principles

Every contribution must preserve these DataChef principles:

1. **Computation is ground truth.** Numerical evidence must come from deterministic analytical tools, not model memory.
2. **Data stays local by default.** Raw dataset rows must not be transmitted to remote inference services.
3. **Investigations are bounded.** Model turns, tool calls, time, memory, and result sizes must have enforceable limits.
4. **Evidence and interpretation are separate.** Model interpretations must remain distinguishable from computed facts.
5. **Investigations are auditable.** Tool calls, arguments, results, failures, and model configuration must be traceable.

## Questions and support

Use GitHub issues for bugs, feature proposals, and scoped technical discussions. Use GitHub Discussions for open-ended questions or ideas when Discussions are enabled.

Thank you for contributing to DataChef and helping people understand their data more reliably.
