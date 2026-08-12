__all__ = ["VLXSeek1_5ForCausalLM", "VLXSeek1_5Config"]


def __getattr__(name):
    if name in __all__:
        try:
            from .language_model.modeling_vlx_seek_1_5 import VLXSeek1_5ForCausalLM, VLXSeek1_5Config
        except ModuleNotFoundError as exc:
            if "qwen3_5" in str(exc):
                raise ImportError(
                    "VLX-Seek 1.5 requires a transformers build that provides "
                    "`transformers.models.qwen3_5`."
                ) from exc
            raise
        return {
            "VLXSeek1_5ForCausalLM": VLXSeek1_5ForCausalLM,
            "VLXSeek1_5Config": VLXSeek1_5Config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
