# Host llama.cpp

llmux can manage an existing Linux `llama-server` alongside Docker profiles.
Select the execution method per profile with `runtime: docker` or `runtime: host`.
Omitting it preserves Docker behaviour. vLLM remains Docker-only.

## Configure your build

Use Linux with Python 3.10+ and kernel pidfd support (5.3+). Install your own
foreground llama-server executable and its libraries; llmux does not build or
install host binaries. Root privileges and systemd are not required.

```yaml
version: 1
profiles:
  - name: custom-host
    backend: llamacpp
    runtime: host
    server_binary: /opt/custom-llama/bin/llama-server
    model_path: /srv/models/model.gguf
    config_name: custom-host
    port: 8080
    gpu_id: "0"
    env_vars:
      LD_LIBRARY_PATH: /opt/custom-llama/lib
```

`server_binary` defaults to `llama-server` on `PATH`. Filesystem paths expand `~`;
relative paths are relative to the llmux project root. The process also runs
from that root. `defaults.llamacpp.runtime` and `defaults.llamacpp.server_binary`
can set defaults for new profiles. Existing Docker profiles can explicitly set
`runtime: docker` to override a host default.

Use the usual config YAML for flags, including custom flags and `extra-args`:

```yaml
# config/llamacpp/custom-host.yaml
alias: custom-host
ctx-size: 4096
n-gpu-layers: 0
```

This example uses CPU inference. Choose a suitable `n-gpu-layers` for your build
to use GPU inference. `gpu_id` sets `CUDA_VISIBLE_DEVICES`; NVIDIA tools are not
required for CPU builds. Other accelerator builds can use their own environment
variables, but llmux's GPU telemetry remains NVIDIA-specific.

The child inherits the launching environment, configured HF settings, and
profile `env_vars`. Managed GPU selection takes precedence. Keep credentials in
the inherited environment or owner-only `.env.common`; do not put them in flags.
Arguments are passed directly to the executable, without shell interpretation.
The UI and `llmux config flags --profile custom-host --json` inspect that binary's
`--help`; custom flags absent from help remain usable.

## Models and preparation

Choose one model source:

- `model_path`: an existing GGUF file. All split GGUF shards must be present;
  llmux launches the first shard. Symlinks, including HF snapshot paths, work.
- `hf_repo` and `hf_file` (or the existing `model_file`/config `model-file`
  fallback): download the selected GGUF and its split shards with the Python
  Hugging Face client. Do not also set `model_path`.

Host `prepare` validates the binary/config and prepares weights without starting
the server. Host `up` performs preparation before launching. Complete cached
snapshots are reused offline. Existing `HF_CACHE_PATH` in `.env.common` maps to
the Hugging Face home directory; otherwise the inherited HF cache settings or
the library's normal user cache apply. `HF_TOKEN`, `PREPARE_MAX_WORKERS`, and
`prepare --max-workers` retain their existing roles. No downloader image is used.
An absent `.env.common` is valid for host profiles.

## Local asset library

Set one or more read-only roots in `profiles.yaml` to catalog an existing model
library. The catalog only reads filenames and filesystem metadata; it never
moves, downloads, hashes, or changes your files.

```yaml
defaults:
  llamacpp:
    library_roots:
      - /srv/llama-cpp
```

`llmux library list` shows catalog entries, and `--kind model`, `mmproj`,
`draft`, or `template` narrows the result. Directory conventions classify GGUF
files under `mmproj/` as projectors and files under `draft/` or `mtp/` as
speculative drafts; other GGUF files are models. `.jinja` files are templates.
Use `--root /path` to inspect a root before adding it to defaults.

Create a host profile from explicit catalog entries:

```bash
uv run llmux profile from-library qwen-vl \
  --model /srv/llama-cpp/models/qwen.gguf \
  --mmproj /srv/llama-cpp/mmproj/qwen-mmproj.gguf \
  --draft /srv/llama-cpp/mtp/qwen-draft.gguf \
  --draft-type draft-mtp \
  --chat-template /srv/llama-cpp/chat-templates/qwen.jinja
```

Each companion is optional and must be selected explicitly. The TUI profile
form has matching Browse controls that select an entry from the configured
catalog. Suggestions are not applied automatically.

For a primary GGUF with embedded MTP heads, omit `draft_model_path`, set
`draft_type: draft-mtp`, and optionally set `mtp_n` to pass
`--spec-draft-n-max`. Leaving `mtp_n` unset uses llama-server's default.

```bash
uv run llmux prepare custom-host
uv run llmux up custom-host
uv run llmux ps --runtime host --json
uv run llmux logs custom-host --tail 100 --no-follow
uv run llmux config flags --profile custom-host --json
uv run llmux env-check --profile custom-host --json
uv run llmux top custom-host
uv run llmux down custom-host
```

The TUI has the same runtime/binary/model controls in profile and quick-setup
forms. Host startup hides Docker image selection. Docker image/build options
such as `up --pull`, `--dev`, and `--tag` are rejected for host profiles.

## Lifecycle and status

Host servers bind to `127.0.0.1` at the profile port. llmux manages the bind
address, port, primary model, metrics, and disabled web UI; config/extra arguments
cannot override these settings. Benchmarking and monitoring use the same HTTP
endpoints as Docker profiles.

A detached supervisor retains process identity, exit status, and rotating logs
under `.runtime/llamacpp/host/<profile>/`. State is private to the current user;
credentials and raw launch arguments are not written into the state file.
Logs rotate at 10 MiB with three backups and remain readable after shutdown.

Closing the CLI/TUI, ending log-follow, or cancelling readiness after a registered
launch leaves the server running. An incomplete launch or cancelled download is
cleaned up. Readiness uses `/v1/models`: the initial 45-second deadline extends
when server logs show progress, up to 30 minutes. A timeout returns non-zero and
leaves the process available for logs and shutdown. An early process exit fails
startup and records the exit code. Repeated `up` on a ready server is a no-op.

`down` sends SIGTERM and escalates to SIGKILL after 30 seconds. Signals target
verified Linux process identities through pidfds; stale state or PID reuse cannot
cause an unrelated process to be stopped. Unreadable/corrupt state is an error,
not evidence that the server is stopped. Concurrent lifecycle operations may
return an operation-in-progress error; retry after the first operation completes.

There is no automatic restart after a crash or reboot, no service installation,
and no adoption of independently launched servers. Stop active profiles before
changing runtime, renaming, deleting, or changing host launch settings. Cloning
copies configuration only. Stopped-profile renames begin a new runtime history;
previous logs remain under the old runtime directory.

Mixed dashboards retain host results when Docker is unavailable and mark Docker
results unknown. Aggregate CLI queries report partial results and exit non-zero
on probe failures. Use `ps --runtime host` or `stats --runtime host` for host-only
queries. Port conflicts always block startup; GPU overlap can be accepted through
the existing CLI `--force` or TUI confirmation. Optional Docker overlap attribution
cannot make host startup depend on Docker being available.
