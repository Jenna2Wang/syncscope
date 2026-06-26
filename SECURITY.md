# Security Policy

## Supported versions

`syncscope` is pre-1.0, so fixes land on the latest released minor version.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Instead, use GitHub's private vulnerability reporting on this repository
(*Security → Report a vulnerability*). Include:

- a description of the issue and its impact,
- steps or a minimal snippet to reproduce it,
- any suggested mitigation.

You can expect an initial acknowledgement within a few days. Once a fix is ready
we will cut a patch release and credit the reporter unless anonymity is
requested.

## Scope

`syncscope` processes numeric arrays and, with the optional extras, media files.
Reports about unsafe handling of untrusted input (e.g. crafted media files via
the optional loaders) are in scope.
