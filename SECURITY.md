# Security

Report vulnerabilities privately via GitHub Security Advisories.

Scope notes:

- `reflex` never sits in a deny path and never executes model-generated
  content (the model generates nothing).
- Policy files are configuration, not code; validate them at load (see
  `app/config.py`) and treat their hosts as trusted input.
- Weights are baked at build time or pulled from your registry over TLS —
  never at request time.
