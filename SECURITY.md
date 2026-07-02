# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in this repository (e.g. unsafe
handling of uploaded audio files, dependency vulnerabilities, or a way to
bypass the true-peak/crest-factor safety constraints in a way that could be
exploited), please report it privately rather than opening a public issue:

- Use GitHub's **[Private vulnerability reporting](../../security/advisories/new)**
  feature for this repository (Settings → Security → enable it if not
  already active), **or**
- Contact the maintainer directly (see repository owner contact info).

Please do not open a public GitHub issue for security reports until a fix
has been released.

## Supported versions

This is a single-line, actively-developed prototype — only the `main`
branch is supported. There are no maintained release branches yet.

| Version | Supported |
| ------- | --------- |
| `main`  | ✅ |

## Automated scanning

This repository runs:
- **CodeQL** (`.github/workflows/codeql.yml`) on every push/PR to `main`
  and on a weekly schedule, for static analysis security scanning.
- **Dependabot** (`.github/dependabot.yml`) for weekly dependency update
  PRs across both Python packages and GitHub Actions.

## Scope note on the Gemini API key

If you enable the optional Gemini decision engine, your `GEMINI_API_KEY` /
`GOOGLE_API_KEY` is read only from the environment (see `.env.example`) and
is never logged, written to disk, or included in any report/JSON output
produced by this project. Do not commit `.env` files — they are excluded
via `.gitignore`.
