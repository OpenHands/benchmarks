---
name: custom-codereview-guide
description: Repository-specific dependency supply-chain review rules.
triggers:
  - /codereview
---

# Repository code review guidance

Read `AGENTS.md` and apply this guide with the general code-review skill.

## Dependency release-age review

For dependency additions or updates, inspect every changed third-party npm or
PyPI version, including transitive lockfile changes. Require at least seven days
since publication of that exact version. Use registry publication/upload metadata
or the relevant artifact's `uv.lock` `upload-time`; a weekly bot schedule, an old
package name, or a passing install is not age evidence.

For an under-seven-day artifact, leave a COMMENT naming the package, version,
publication time, and source, and defer approval until it ages out or a maintainer
explicitly approves a narrowly documented exception. If publication time is
unavailable, ask for that evidence. Existing named first-party exemptions apply
only to those packages, not their third-party dependencies or blanket approval
of the lockfile. Apply this check before any dependency-bump fast path.

Flag removal or weakening of seven-day uv/Dependabot guards or new broad
exemptions. Dependabot security updates bypass cooldown, so review them under
the explicit maintainer-exception rule in `AGENTS.md`; agents cannot approve
their own bypass.
