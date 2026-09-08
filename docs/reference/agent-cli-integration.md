# AI-agent CLI integration reference

`llmux` is the supported integration boundary for a program or AI agent that
needs to discover, observe, or control llmux-managed servers. This reference is
consumer-neutral: use it for an automation, an IDE integration, an agent, or a
dashboard.

## Operating contract

Run the installed `llmux` executable as the same WSL/Linux user that owns the
target llmux project's profiles. Use its documented JSON-capable commands and
their exit codes as the API contract. Do not parse TUI or terminal-table output,
and do not read `.runtime` state or environment files directly.

The CLI resolves a project from its current directory when that directory is an
llmux checkout. To intentionally target a different checkout, set `LLMUX_ROOT`
to that project's root before invoking the command. Profiles and host runtime
state are user-local; do not assume that another user sees or can manage them.

Examples below use an installed `llmux` command. A source-checkout workflow may
substitute `uv run llmux`.

## Discovery and status

Use these non-mutating commands for llmux-managed inventory and status:

```bash
llmux ps --runtime host --json
llmux ps --runtime host --running --json
llmux profile list --json
llmux profile show NAME --json
```

`ps --json` emits an array. For host status, the fields most useful to a
consumer are:

| Field | Meaning |
| --- | --- |
| `profile` | The llmux profile name: the lifecycle target. |
| `runtime` | `host` or `docker`. |
| `running` | Whether llmux currently considers the server process/container running. |
| `ready` | Whether the server has passed its readiness check. |
| `status` | Human-readable runtime status, useful for display. |
| `port` | Local listening port when configured; use this to derive the endpoint. |
| `pid` | Host process ID, when applicable and known. |
| `executable` | Host server executable, when applicable. |
| `exit_code` | Recorded host process exit code, when available. |
| `error` | Per-profile probe or runtime error text, when available. |

The reported port defines the local OpenAI-compatible base URL:

```text
http://127.0.0.1:PORT/v1
```

Use `profile show NAME --json` when the complete saved profile record is
needed. It intentionally redacts sensitive profile environment values.

## Failure handling

Exit status is part of the protocol. A non-zero result means a command or a
probe failed, even if stdout contains valid JSON rows from another backend or
profile. Preserve and parse stdout JSON when present, but also surface stderr
and each row's `error`; do not reinterpret an empty result as "no servers."

For example, an aggregate status query can return partial rows while Docker is
unavailable. Use a runtime filter such as `--runtime host` when that is the
only authority required; it avoids making a host-only view depend on a Docker
probe. For diagnostics, add `--log-level DEBUG` before the subcommand.

## Observability

Take a finite log snapshot rather than attaching a machine workflow to a
stream:

```bash
llmux logs NAME --tail 200 --no-follow
```

Poll token throughput with one finite tick:

```bash
llmux stats --runtime host --once --json
```

`stats --json` writes one compact JSON array per tick (NDJSON). `--once` first
collects a baseline, waits one interval, prints one rate tick, and exits. The
records include `backend`, `profile`, `port`, `runtime`,
`prompt_tok_per_s`, and `generation_tok_per_s`. Each record also contains
`sampled_at` (UTC), `sample_interval_seconds`, `availability`, and `error`.
`availability` is `available`, `not_ready`, or `unavailable`; unavailable
observations retain their identity and null rates. A partial probe still writes
the valid rows, then exits non-zero.

For a finite, structured log result, request a snapshot explicitly:

```bash
llmux logs NAME --tail 200 --no-follow --json
```

This writes `{ "status": "ok", "result": { ... } }`, where `result` includes
the profile identity, runtime, tail, availability, and `lines`. Errors use
`{ "status": "error", "error": { "code", "message" } }`; the stable codes
are `PROFILE_NOT_FOUND`, `FINITE_SNAPSHOT_REQUIRED`, and `LOGS_UNAVAILABLE`.
JSON log snapshots never follow a stream.

Do not use `llmux top` for machine integration. It is a terminal-rendered live
monitor, not a JSON API. Query `ps`, `stats`, and the server's reported
OpenAI-compatible endpoint instead.

## Lifecycle controls

The following commands change state and should be presented separately from
discovery actions:

```bash
llmux prepare NAME
llmux up NAME
llmux down NAME
```

`prepare` renders runtime files and obtains required weights without starting
the server. `up` prepares and starts it; `down` stops it. Control profiles and
configs through `llmux profile ...` and `llmux config ...` only—never by
editing runtime artifacts.

Host runtime is supported only for llama.cpp profiles; vLLM is Docker-only.
Some Docker-only options are rejected for host profiles. Lifecycle operations
are locked, so a concurrent action can report an operation-in-progress error;
wait for the first operation to finish and retry. Profiles must be stopped
before edits that affect a running profile, and before profile renames or
deletes. Config rename/delete likewise requires any referencing profile to be
stopped.

When `up`, `prepare`, or `down` is called with `--json` while that profile is
busy, it exits with code `75` and writes an error envelope whose code is
`OPERATION_IN_PROGRESS`. The error includes the requested action, current
operation/state when safely known, `retry_after_seconds` (or `null` when it
cannot be determined), and retry guidance. llmux never retries lifecycle
actions automatically.

## Creating profiles and configs

Creation is distinct from discovery and is always mutating. Use the CLI rather
than writing `profiles.yaml` or config YAML directly:

```bash
llmux profile new NAME --backend vllm --model ORGANIZATION/MODEL
llmux profile quick-setup ORGANIZATION/MODEL
llmux config new NAME --backend vllm --model ORGANIZATION/MODEL
llmux config edit NAME --set KEY=VALUE
```

Profile and config mutations accept `--json`. On success they return
`{ "status": "ok", "result": { "profile" | "config": ... } }` with the
authoritative saved record (including redacted profile environment values).
Existing `list` and `show` JSON shapes remain unchanged.

For local llama.cpp assets, first inspect the read-only catalog, then explicitly
select the primary model and any companions:

```bash
llmux library list --json
llmux profile from-library NAME --model MODEL_PATH --mmproj MMPROJ_PATH \
  --draft DRAFT_PATH --draft-type draft-mtp --chat-template TEMPLATE_PATH
```

Every companion is optional, but selection is explicit: do not infer a
projector, speculative draft, or chat template from nearby files. A primary
GGUF with embedded MTP heads is represented on the profile by `draft_type:
draft-mtp` with no `draft_model_path`; optionally set `mtp_n` for its
draft-token limit. `profile quick-setup --backend llamacpp` supports host or
Docker llama.cpp setup from an HF repository/file or a local model path.

The `config` group provides `list`, `show`, `new`, `edit`, `clone`, `rename`,
`flags`, `from-recipe`, and `delete`; use those commands for all config
management. Pair creation with a subsequent `profile show NAME --json` and
`ps --json` rather than assuming a requested name or port was accepted.

## Managed and unmanaged servers

llmux discovers and manages only profiles it owns. It does not adopt servers
launched independently. If a consumer also discovers unmanaged servers, it
must use a separate discovery mechanism, keep that data separate, and label
the source clearly. llmux remains the sole authority for lifecycle actions on
llmux-managed profiles.

## Security and portability

Do not embed machine-specific paths, credentials, tokens, or profile
environment values in an integration. Treat profile names, model identifiers,
and ports as data returned by the CLI. Keep secrets in the launching user's
environment or that user's protected llmux configuration, and avoid logging
them when forwarding command output.
