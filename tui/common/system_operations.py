from __future__ import annotations

from pathlib import Path

from tui.common import profile_store
from tui.common.env import validate_common_env
from tui.common.prepare import stream_pull


async def collect_events(events) -> tuple[int, list[str]]:
    rc = -1
    lines: list[str] = []
    async for kind, data in events:
        if kind == "rc":
            rc = int(data)
        else:
            lines.append(str(data))
    return rc, lines


async def pull_image(image_ref: str) -> tuple[int, list[str]]:
    return await collect_events(stream_pull(image_ref))


async def remove_image(image_ref: str, *, force: bool = False) -> tuple[int, str]:
    from tui.common.docker import run_command

    args = ["docker", "rmi"]
    if force:
        args.append("--force")
    args.append(image_ref)
    return await run_command(*args, timeout=120)


async def build_dev_image(
    backend: str,
    *,
    repo_url: str,
    branch: str,
    custom_tag: str,
    official: bool = False,
    cuda_arch: str = "",
    multi_arch: bool = False,
) -> tuple[int, list[str]]:
    if backend == "vllm":
        from tui.backends.vllm.backend_runtime import _stream_build_dev_image

        events = _stream_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            use_official=official,
        )
    elif backend == "llamacpp":
        from tui.backends.llamacpp.backend_runtime import _stream_build_dev_image

        events = _stream_build_dev_image(
            branch,
            repo_url=repo_url,
            custom_tag=custom_tag,
            cuda_arch=cuda_arch,
            use_multi_arch=multi_arch,
        )
    else:
        raise ValueError(f"invalid backend {backend!r}")
    return await collect_events(events)


def environment_status(common_env: Path, *, profile: str = "") -> tuple[bool, list[str]]:
    profiles = [p for bk in ("vllm", "llamacpp") for p in profile_store.list_profiles(bk)]
    if profile:
        profiles = [p for p in profiles if p.name == profile]
        if not profiles:
            return False, [f"Profile not found: {profile}"]
    messages = []
    ok = True
    if not profiles or any(p.runtime == "docker" for p in profiles):
        ok, messages = validate_common_env(common_env)
    for p in profiles:
        if p.runtime == "host":
            from tui.backends.llamacpp.host_runtime import executable, model_paths
            from tui.backends.llamacpp.backend import load_config
            try:
                binary = executable(p)
                model_paths(p, load_config(p.config_name or p.name).params)
                messages.append(f"Host {p.name}: {binary}")
            except (OSError, ValueError, RuntimeError) as exc:
                ok = False
                messages.append(f"Error: host {p.name}: {exc}")
    return ok, messages


def render_backend_envs(backend: str) -> tuple[list[Path], list[str]]:
    rendered: list[Path] = []
    failures: list[str] = []
    for profile in profile_store.list_profiles(backend):
        try:
            rendered.append(
                profile_store.render_env_for_profile(profile.name, profile.backend)
            )
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{profile.name}: {exc}")
    return rendered, failures
