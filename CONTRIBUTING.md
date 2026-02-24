# Contributing to Burn

Thank you for your interest in contributing! We welcome contributions from the community, but we
hold every contributor to a high standard of ownership and quality. Please read this guide carefully
before opening a pull request.

## The Golden Rule: Own Your Work

Every line of code you submit must be code you **understand and can defend**. If you cannot explain
why a change was made, how it works, and what trade-offs were considered, the PR will be closed.

We recognize that AI coding assistants (Copilot, Cursor, Claude, ChatGPT, etc.) are part of modern
development workflows, and we do not ban their use. However, **you are the author, not your AI
agent**. Using an AI tool does not absolve you of the responsibility to understand, verify, and
stand behind every change in your PR.

## AI-Assisted Contributions Policy

### What we expect

- You have **read and understood** every line of code in your PR before submitting it.
- You can explain the **rationale** behind each change if asked during review.
- You have **tested your changes** locally and confirmed they work as intended.
- You have reviewed AI-generated code for correctness, style consistency, and relevance to the issue
  at hand.
- You treat AI tools as assistants, not authors. The intellectual ownership is yours.

### What will get your PR closed

- Submitting large, AI-generated diffs with no evidence of human review or comprehension.
- Inability to answer reviewer questions about your own code.
- PRs that "shotgun" changes across unrelated files with no clear rationale.
- Superficial or cosmetic-only changes disguised as meaningful contributions (e.g., mass
  reformatting, trivial comment rewording, renaming without purpose).
- Copy-pasting AI output that introduces hallucinated APIs, incorrect logic, or irrelevant code.
- Repeatedly submitting low-quality PRs that waste reviewer time.

### Reviewer discretion

Maintainers reserve the right to ask clarifying questions about any part of your PR to verify
understanding. If your answers suggest you do not understand the changes you submitted, the PR will
be closed. Repeat offenses may result in future PRs being deprioritized or the contributor being
blocked.

## Before You Open a PR

1. **Check for an existing issue.** If there isn't one, open an issue first to discuss the approach.
   Do not surprise reviewers with unsolicited large refactors.
2. **Keep it focused.** One PR should address one concern. Do not bundle unrelated changes together.
3. **Read the codebase.** Understand the architecture and conventions already in place. PRs that
   ignore existing patterns will be rejected. The
   [Contributor Book](https://burn.dev/contributor-book/) covers architecture, environment setup,
   and guides for common tasks.
4. **Run validation.** Run `cargo run-checks` before you submit. This runs formatting, linting, and
   the full test suite. All checks must pass.
5. **Write a clear PR description.** Explain _what_ you changed, _why_ you changed it, and _how_ you
   verified it works. Link the relevant issue.

## PR Requirements

Every pull request must include:

- **A descriptive title** that summarizes the change (not "Fix stuff" or "Update code").
- **A description** that covers:
  - The problem being solved or feature being added.
  - The approach taken and why.
  - Any trade-offs or alternatives considered.
  - How the change was tested.
- **Passing CI checks.** Do not ask reviewers to look at a red build.
- **Minimal scope.** No drive-by changes. If you spot an unrelated issue, open a separate PR for it.

## Code Quality Standards

- Follow the existing code style and project conventions.
- Write idiomatic Rust. If you are unfamiliar with Rust conventions, please study the existing
  codebase before contributing.
- Keep dependencies minimal. Do not introduce new crates without discussion.
- Document public APIs. Non-trivial logic should have comments explaining _why_, not just _what_.
- Prefer clarity over cleverness.
- Bug fixes must include a regression test.

## Review Process

- Maintainers will review PRs as time allows. Please be patient.
- Be responsive to feedback. If changes are requested, address them or explain your reasoning.
- Do not force-push to rewrite history during an active review without notice.
- If a PR goes stale for more than 14 days without response from the author, it may be closed.

## A Note on Contribution Quality vs. Quantity

We value **meaningful contributions** over PR count. One well-crafted, well-understood PR is worth
more than ten sloppy ones. We are a small project with limited reviewer bandwidth, and every
low-quality PR takes time away from actual development. Please respect that time.

If your goal is to pad a contribution graph rather than to genuinely improve this project, this is
not the repo for you.

## Getting Help

If you're unsure about an approach, open an issue or start a discussion first. We are happy to guide
contributors who are genuinely engaged. Asking questions is a sign of strength, not weakness.

For questions and discussions, join us on [Discord](https://discord.gg/uPEBbYYDB6).

---

By submitting a pull request, you confirm that you have read these guidelines, that you understand
and can explain the code you are submitting, and that you are the intellectual owner of the
contribution.
