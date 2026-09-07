"""Runtime-aware status collection; failed probes never imply stopped."""
from __future__ import annotations

from . import profile_store


async def statuses(backend: str | None = None, runtime: str | None = None):
    if runtime not in (None, "docker", "host"):
        raise ValueError("runtime must be docker or host")
    has_host = any(p.runtime == "host" for p in profile_store.list_profiles("llamacpp"))
    results = []
    errors = []
    for bk in ([backend] if backend else ["vllm", "llamacpp"]):
        profiles = [p for p in profile_store.list_profiles(bk) if runtime is None or p.runtime == runtime]
        if not profiles and (has_host or runtime == "host"):
            continue
        if bk == "llamacpp":
            from tui.backends.llamacpp.backend_runtime import get_container_statuses
            rows = await get_container_statuses(runtime=runtime, partial=True)
            for row in rows:
                if row.error:
                    errors.append(f"{bk}/{row.profile_name}: {row.error}")
        else:
            from tui.backends.vllm.backend_runtime import get_container_statuses
            try:
                rows = await get_container_statuses()
            except RuntimeError as exc:
                from tui.backends.llamacpp.backend_runtime import ContainerStatus
                errors.append(f"{bk}: {exc}")
                rows = [ContainerStatus(profile_name=p.name, container_name=p.container_name or p.name,
                    status_text="unknown", port=str(p.port), gpu_id=p.gpu_id,
                    model=p.model_id, error=str(exc)) for p in profiles]
        results.extend((bk, row) for row in rows)
    return results, errors


async def dashboard_rows():
    from .adapter import DashboardRow
    rows, errors = await statuses()
    result = []
    for bk, status in rows:
        if bk == "llamacpp":
            from tui.backends.llamacpp.backend import load_profile
        else:
            from tui.backends.vllm.backend import load_profile
        profile = load_profile(status.profile_name)
        profile.running = status.running
        result.append(DashboardRow(
            backend=bk, profile_name=status.profile_name, container_name=status.container_name,
            port=int(status.port) if status.port else profile.port, running=status.running,
            model=status.model, detail="", gpu_id=status.gpu_id, raw=profile,
            runtime=getattr(status, "runtime", "docker"), pid=getattr(status, "pid", None),
            status_text=status.status_text))
    return result, errors


async def require_stopped(profiles) -> None:
    """Mutation guard for config operations spanning both runtime types."""
    from . import docker
    host_profiles = [p for p in profiles if getattr(p, "runtime", "docker") == "host"]
    for p in host_profiles:
        from tui.backends.llamacpp.host_process import require_stopped as host_stopped
        host_stopped(p.name)
    docker_profiles = [p for p in profiles if getattr(p, "runtime", "docker") == "docker"]
    if docker_profiles:
        running = await docker.running_container_names()
        for p in docker_profiles:
            if (p.container_name or p.name) in running:
                raise ValueError(f"Profile '{p.name}' is running; stop it first")
