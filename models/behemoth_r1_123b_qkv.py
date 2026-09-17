from .base import ModelQuantConfig


class _BehemothR1_123BQkvConfig(ModelQuantConfig):
    """All linears in NVFP4 except embeddings and lm_head. Targets ~65 GiB.

    This exists because behemoth_r1_123b_q does not load. vLLM fuses q/k/v into
    one QKVParallelLinear and requires a single precision across every shard, so
    quantizing q_proj alone produces an export that passes every per-module
    check and then dies at load with "Detected some but not all shards of
    model.layers.0.self_attn.qkv_proj are quantized".

    That leaves exactly two servable choices for the attention block: all of
    q/k/v in BF16 (behemoth_r1_123b, 86 GiB) or all of them in NVFP4 (here,
    65.3 GiB). There is no middle option, however attractive the parameter
    arithmetic makes one look.

    k_proj and v_proj are only 2.21B params combined, so this saves just 3.2 GB
    over the unservable q-only export -- the size was never the argument. The
    argument is that 65.3 GiB loads on a single 96 GB card, which removes
    tensor-parallel all-reduce entirely (see Behemoth-123B_v2_R1.md section 3).

    The KV *cache* is still BF16 and must stay that way. Quantizing the k/v
    projection weights is a different thing from quantizing the cache they
    write into, and only the former happens here.
    """

    def get_model_cls(self):
        from transformers import MistralForCausalLM

        return MistralForCausalLM

    def get_all_quant_overrides(self) -> dict:
        return dict(self.extra_quant_overrides)


BehemothR1_123BQkvConfig = _BehemothR1_123BQkvConfig(
    model_id="/media/fmodels/TheDrummer/Behemoth-R1-123B-v2",
    trust_remote_code=False,
    streaming=False,
    base_quant_cfg="NVFP4_DEFAULT_CFG",
    extra_quant_overrides={
        # BF16 KV cache. No attention BMM or softmax quantization, ever.
        "*q_bmm_quantizer": {"enable": False},
        "*[kv]_bmm_quantizer": {"enable": False},
        "*softmax_quantizer": {"enable": False},

        # Untied 32k-vocab embeddings and head stay BF16. These are the only
        # exclusions, which is what keeps every fused group uniform.
        "*embed_tokens*weight_quantizer": {"enable": False},
        "*embed_tokens*input_quantizer": {"enable": False},
        "*lm_head*weight_quantizer": {"enable": False},
        "*lm_head*input_quantizer": {"enable": False},

        # Left enabled -> NVFP4 W4A4 on all seven linears per layer.
    },
)
