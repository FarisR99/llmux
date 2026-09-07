"""LlamacppAdapter — llamacpp backend 를 통합 Dashboard 규격으로 노출."""

from __future__ import annotations

from tui.backends.llamacpp import backend as lbackend
from tui.common.adapter import DashboardRow
from tui.common.docker import container_is_running


class LlamacppAdapter:
    name = "llamacpp"
    display_name = "llama.cpp"
    accent_color = "#16a34a"   # green-600

    def rows(self, running: set[str]) -> list[DashboardRow]:
        out: list[DashboardRow] = []
        profiles = lbackend.list_profiles(running=running)
        for p in profiles:
            config = lbackend.load_config(p.config_name or p.name)
            alias = config.get("alias", "")
            model = alias.strip() if isinstance(alias, str) else ""
            if not model:
                model = p.hf_repo or p.model_file or ""
            host_state = None
            if p.runtime == "host":
                from .host_process import status
                host_state = status(p.name)
            detail = ""
            if p.model_size_gb is not None:
                detail = f"{p.model_size_gb:.1f} GB"
            elif p.hf_file:
                detail = p.hf_file.split("/")[-1]
            out.append(
                DashboardRow(
                    backend=self.name,
                    profile_name=p.name,
                    container_name="" if p.runtime == "host" else p.container_name or p.name,
                    port=p.port or None,
                    running=host_state["running"] if host_state else container_is_running(running, p.container_name or p.name),
                    runtime=p.runtime,
                    pid=host_state["pid"] if host_state else None,
                    status_text=host_state["status"] if host_state else "",
                    model=model,
                    detail=detail,
                    gpu_id=p.gpu_id or "",
                    raw=p,
                )
            )
        return out
