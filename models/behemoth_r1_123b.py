from .base import ModelQuantConfig


class _BehemothR1_123BConfig(ModelQuantConfig):
    """Dense Mistral-Large-2411 finetune. NVFP4 on MLP + o_proj, everything else BF16.

    Deliberately does not inherit COMMON_QUANT_OVERRIDES: that blanket-disables
    every ``*self_attn*`` quantizer (which would also exclude o_proj) and turns
    on FP8 KV. Both are wrong here, and relying on later patterns to win over
    earlier ones inside a single quant_cfg is fragile.
    """

    def get_model_cls(self):
        from transformers import MistralForCausalLM

        return MistralForCausalLM

    def get_all_quant_overrides(self) -> dict:
        return dict(self.extra_quant_overrides)


BehemothR1_123BConfig = _BehemothR1_123BConfig(
    model_id="/media/fmodels/TheDrummer/Behemoth-R1-123B-v2",
    trust_remote_code=False,
    # 229 GB BF16 fits in 4x96 GB, so accelerate can hold the whole model.
    streaming=False,
    # NVIDIA's own MLP + o_proj scope, rather than hand-rolling it on top of
    # NVFP4_DEFAULT_CFG. The overrides below are then belt-and-braces.
    base_quant_cfg="NVFP4_OMLP_ONLY_CFG",
    extra_quant_overrides={
        # QKV stays BF16. These carry the widest activation dynamic range per
        # parameter, and k/v are only 12.6M params each here (GQA, 8 KV heads),
        # so quantizing them buys ~1.8% of the model for the highest risk.
        "*q_proj*weight_quantizer": {"enable": False},
        "*q_proj*input_quantizer": {"enable": False},
        "*k_proj*weight_quantizer": {"enable": False},
        "*k_proj*input_quantizer": {"enable": False},
        "*v_proj*weight_quantizer": {"enable": False},
        "*v_proj*input_quantizer": {"enable": False},

        # BF16 KV cache. No attention BMM or softmax quantization, ever.
        "*q_bmm_quantizer": {"enable": False},
        "*[kv]_bmm_quantizer": {"enable": False},
        "*softmax_quantizer": {"enable": False},

        # Untied 32k-vocab embeddings and head stay BF16.
        "*embed_tokens*weight_quantizer": {"enable": False},
        "*embed_tokens*input_quantizer": {"enable": False},
        "*lm_head*weight_quantizer": {"enable": False},
        "*lm_head*input_quantizer": {"enable": False},

        # Left enabled -> NVFP4 W4A4: gate_proj, up_proj, down_proj, o_proj.
    },
)
