# Security policy

## Supported versions

HamLeT is currently pre-1.0. Security fixes are made on the latest release
line; users should upgrade to the newest published version before reporting a
problem.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| older development snapshots | No |

## Reporting a vulnerability

Please do not open a public issue for a vulnerability. Use GitHub's **Report a
vulnerability** link in the repository Security tab to open a private security
advisory. Include the affected version, operating system, reproduction steps,
and the impact you observed.

HamLeT's browser interface is intended to bind to localhost. Treat using
`--host` to expose it to another network as an advanced deployment: the
development server does not provide authentication or TLS and must not be
published directly to the internet.
