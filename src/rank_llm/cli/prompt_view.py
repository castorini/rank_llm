from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


TEMPLATES = files("rank_llm.rerank.prompt_templates")


def list_prompt_templates() -> list[dict[str, Any]]:
    catalog = []
    for path in sorted(Path(str(TEMPLATES)).glob("*.yaml")):
        template = load_prompt_template(path.name)
        placeholders = sorted(_collect_placeholders(template))
        catalog.append(
            {
                "name": path.stem,
                "path": str(path),
                "method": template.get("method", ""),
                "placeholders": placeholders,
            }
        )
    return catalog


def load_prompt_template(name_or_path: str) -> dict[str, Any]:
    candidate_path = Path(name_or_path)
    path = candidate_path
    if not candidate_path.exists():
        template_name = (
            name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"
        )
        path = Path(str(TEMPLATES / template_name))
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_prompt_template_view(name_or_path: str) -> dict[str, Any]:
    template = load_prompt_template(name_or_path)
    return {
        "name": Path(name_or_path).stem,
        "template": template,
        "placeholders": sorted(_collect_placeholders(template)),
    }


def build_rendered_prompt_view(
    name_or_path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    template = load_prompt_template(name_or_path)
    method = template["method"]
    query_text = _query_text(payload["query"])
    candidates = payload["candidates"]

    if method == "singleturn_listwise":
        user_content = ""
        if template.get("prefix"):
            user_content += template["prefix"].format(
                query=query_text,
                num=len(candidates),
            )
        for index, candidate in enumerate(candidates, start=1):
            user_content += template["body"].format(
                rank=index,
                candidate=_candidate_text(candidate),
                score=_candidate_score(candidate),
            )
        if template.get("suffix"):
            user_content += template["suffix"].format(
                query=query_text,
                num=len(candidates),
            )
        messages = _with_system(template, [{"role": "user", "content": user_content}])
    elif method == "multiturn_listwise":
        messages = _with_system(template, [])
        if template.get("prefix_user"):
            messages.append(
                {
                    "role": "user",
                    "content": template["prefix_user"].format(
                        query=query_text,
                        num=len(candidates),
                    ),
                }
            )
        if template.get("prefix_assistant"):
            messages.append(
                {
                    "role": "assistant",
                    "content": template["prefix_assistant"].format(
                        query=query_text,
                        num=len(candidates),
                    ),
                }
            )
        for index, candidate in enumerate(candidates, start=1):
            messages.append(
                {
                    "role": "user",
                    "content": template["body_user"].format(
                        rank=index,
                        candidate=_candidate_text(candidate),
                    ),
                }
            )
            if template.get("body_assistant"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": template["body_assistant"].format(rank=index),
                    }
                )
        if template.get("suffix_user"):
            messages.append(
                {
                    "role": "user",
                    "content": template["suffix_user"].format(
                        query=query_text,
                        num=len(candidates),
                    ),
                }
            )
    elif method == "pointwise":
        candidate = candidates[0]
        messages = [
            {
                "role": "user",
                "content": template["body"].format(
                    query=query_text,
                    doc_content=_candidate_text(candidate),
                ),
            }
        ]
    elif method == "pairwise":
        candidate_1 = candidates[0]
        candidate_2 = candidates[1] if len(candidates) > 1 else candidates[0]
        messages = [
            {
                "role": "user",
                "content": template["body"].format(
                    query=query_text,
                    doc1=_candidate_text(candidate_1),
                    doc2=_candidate_text(candidate_2),
                ),
            }
        ]
    elif method == "rankfid":
        lines = [
            template["query"].format(query=query_text),
        ]
        for candidate in candidates:
            lines.append(template["text"].format(text=_candidate_text(candidate)))
        messages = [{"role": "user", "content": "\n".join(lines)}]
    else:
        raise ValueError(f"Unsupported template method: {method}")

    return {
        "name": Path(name_or_path).stem,
        "method": method,
        "messages": messages,
        "inputs": {"query": query_text, "candidate_count": len(candidates)},
    }


def render_prompt_catalog_text(catalog: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{entry['name']} [{entry['method']}] {entry['path']}" for entry in catalog
    )


def render_prompt_template_text(view: dict[str, Any]) -> str:
    template = view["template"]
    lines = [
        f"name: {view['name']}",
        f"method: {template.get('method', '')}",
    ]
    for key, value in template.items():
        if key == "method":
            continue
        lines.append(f"[{key}]")
        lines.append(str(value))
    return "\n".join(lines)


def render_rendered_prompt_text(view: dict[str, Any]) -> str:
    lines = [f"name: {view['name']}", f"method: {view['method']}"]
    for message in view["messages"]:
        lines.append(f"[{message['role']}]")
        lines.append(message["content"])
    return "\n".join(lines)


def _with_system(
    template: dict[str, Any],
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_message = template.get("system_message")
    if system_message:
        return [{"role": "system", "content": system_message}, *messages]
    return messages


def _query_text(query: Any) -> str:
    if isinstance(query, dict):
        return str(query["text"])
    return str(query)


def _candidate_text(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if "text" in candidate:
        return str(candidate["text"])
    doc = candidate["doc"]
    if isinstance(doc, str):
        return doc
    for key in ("text", "segment", "contents", "content", "body", "passage"):
        if key in doc:
            return str(doc[key])
    return str(doc)


def _candidate_score(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return f"{candidate.get('score', 0.0):.3f}"
    return "0.000"


def _collect_placeholders(template: dict[str, Any]) -> set[str]:
    placeholders: set[str] = set()
    for value in template.values():
        if not isinstance(value, str):
            continue
        current = ""
        inside = False
        for character in value:
            if character == "{":
                inside = True
                current = ""
                continue
            if character == "}" and inside:
                placeholders.add(current)
                inside = False
                continue
            if inside:
                current += character
    return placeholders
