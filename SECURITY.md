# Security Policy

## Supported versions

Security fixes are applied to the latest minor release.

## Reporting a vulnerability

Please do **not** open a public issue for security problems. Use the repository's
**Security** tab on GitHub (*Report a vulnerability*, GitHub's private vulnerability
reporting) and include the version affected, a description of the issue and its impact,
and steps to reproduce. You will receive an acknowledgement within 5 working days, and a
fixed release with credit to the reporter (unless you prefer anonymity).

## Scope notes

- `agent-chaos-engineering` deliberately injects faults. Injection only ever happens through an
  explicitly constructed `ChaosInjector`; a way for faults to reach code that did not
  opt in, or to fire while `enabled=False`, is a bug we want to hear about.
- Never run chaos plans against production systems that have real side effects
  (payments, emails, actuators) without a sandbox: `ToolError` raises inside your tools
  and `Miscoordination` reroutes real messages.
- `agent-chaos run --scenario module:function` imports and executes code from the
  current working directory by design. Only run it in directories you trust.
