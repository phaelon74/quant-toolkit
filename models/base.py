from dataclasses import dataclass, field
from typing import Callable


# Shared quant overrides: disable attention weights/inputs, keep FP8 KV cache.
COMMON_QUANT_OVERRIDES = {
    "*self_attn*weight_quantizer": {"enable": False},
    "*self_attn*input_quantizer": {"enable": False},
    "*self_attn*q_bmm_quantizer": {"enable": False},
    "*self_attn*softmax_quantizer": {"enable": False},
    "*[kv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "enable": True},
}


@dataclass
class ModelQuantConfig:
    model_id: str
    trust_remote_code: bool = False
    streaming: bool = False
    extra_quant_overrides: dict = field(default_factory=dict)
    extra_mtp_prefixes: list = field(default_factory=list)
    # Name of the mtq preset to start from, e.g. "NVFP4_OMLP_ONLY_CFG".
    base_quant_cfg: str = "NVFP4_DEFAULT_CFG"

    def get_model_cls(self):
        """Return explicit model class, or None for AutoModelForCausalLM."""
        return None

    def get_base_quant_cfg(self) -> dict:
        """Resolve base_quant_cfg against modelopt, with a clear error if absent."""
        import modelopt.torch.quantization as mtq

        cfg = getattr(mtq, self.base_quant_cfg, None)
        if cfg is None:
            available = sorted(n for n in dir(mtq) if n.startswith("NVFP4_"))
            raise SystemExit(
                f"{self.base_quant_cfg} is not available in the installed ModelOpt.\n"
                f"NVFP4 presets found: {available}"
            )
        return cfg

    def register_moe(self):
        """Register MoE QuantModules if needed. Override in subclasses."""
        pass

    def get_all_quant_overrides(self) -> dict:
        overrides = dict(COMMON_QUANT_OVERRIDES)
        overrides.update(self.extra_quant_overrides)
        return overrides
