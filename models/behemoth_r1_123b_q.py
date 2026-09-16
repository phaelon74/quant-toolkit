from .base import ModelQuantConfig


class _BehemothR1_123BQConfig(ModelQuantConfig):
    """Same as behemoth_r1_123b, plus q_proj in NVFP4. Targets ~68 GiB.

    q_proj is 151M params per layer -- as large as o_proj, and 12x k_proj or
    v_proj, which are tiny here because of GQA with 8 KV heads. Leaving all of
    QKV in BF16 therefore costs 26.6 GB, of which 26.6 GB is Q. Quantizing it
    drops the checkpoint from 86 GiB to ~68 GiB, which is what lets the model
    load on a single 96 GB card and skip tensor-parallel all-reduce entirely.

    Built up from NVFP4_DEFAULT_CFG rather than NVFP4_OMLP_ONLY_CFG: the
    OMLP preset never creates q_proj quantizers, so no override can switch them
    back on. Starting from the everything-preset and subtracting is the only way
    to be sure q_proj really is in scope.

    k_proj and v_proj stay BF16. They feed the KV cache, they carry the widest
    activation range per parameter, and together they are only 4.4 GB -- the
    worst size-to-risk ratio in the model.
    """

    def get_model_cls(self):
        from transformers import MistralForCausalLM

        return MistralForCausalLM

    def get_all_quant_overrides(self) -> dict:
        return dict(self.extra_quant_overrides)


BehemothR1_123BQConfig = _BehemothR1_123BQConfig(
    model_id="/media/fmodels/TheDrummer/Behemoth-R1-123B-v2",
    trust_remote_code=False,
    streaming=False,
    base_quant_cfg="NVFP4_DEFAULT_CFG",
    extra_quant_overrides={
        # k/v stay BF16.
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

        # Left enabled -> NVFP4 W4A4: q_proj, o_proj, gate_proj, up_proj, down_proj.
    },
)
