from pathlib import Path

from huggingface_hub import try_to_load_from_cache


def resolve_model_path(name_or_path: str) -> str:
    """Prefer a local path, then a cached Hugging Face snapshot."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return str(candidate)

    cached_config = try_to_load_from_cache(name_or_path, "config.json")
    if isinstance(cached_config, str):
        return str(Path(cached_config).parent)

    return name_or_path
