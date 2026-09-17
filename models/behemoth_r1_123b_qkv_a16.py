from .base import ModelQuantConfig


class _BehemothR1_123BQkvA16Config(ModelQuantConfig):
    """As behemoth_r1_123b_qkv, but q/k/v keep BF16 activations. ~65 GiB.

    NVFP4 weights do not oblige NVFP4 activations. Dropping just the q/k/v input
    quantizers gives weight-only NVFP4 there and leaves everything else W4A4,
    which threads a needle the other adapters cannot:

      - It is legal. All three of q/k/v carry the same scheme, so the fused
        QKVParallelLinear is uniform. The illegal thing was never "4-bit
        activations on k/v", it was disagreement between shards.
      - It removes the reason k/v were unwanted in scope at all. k_proj and
        v_proj have the widest activation range per parameter in the model;
        their *weights* were never the worry. This quantizes the weights, which
        is where the 3.2 GB is, and leaves the activations alone.
      - It costs nothing on disk. Weight bytes are identical; only the per-module
        input_scale scalars disappear.

    What it costs is math throughput. Blackwell's FP4 tensor cores need both
    operands in FP4, so these GEMMs fall back to dequantize-and-BF16 and keep
    only the bandwidth win. q/k/v are ~12% of linear FLOPs per layer, which is
    close to free during bandwidth-bound decode and real during prefill.

    o_proj deliberately stays W4A4. Its input is the attention output, a
    different tensor from the one q/k/v read, and the 86 GiB export already
    measured o_proj W4A4 as cheap. There is no reason to widen this further than
    the three projections whose activation quantization is actually in question.

    Two things to establish before trusting an export from this config:
      1. That ModelOpt 0.46 describes a mixed W4A4/weight-only checkpoint in a
         way vLLM reads correctly -- it may need two config_groups, and the
         exporter has only ever been observed emitting one here.
      2. That prefill throughput does not regress more than the accuracy gain is
         worth.
    Run tools/verify_nvfp4_export.py first; it reports per-kind A4 vs A16 and
    fails a fused group that mixes the two.
    """

    def get_model_cls(self):
        from transformers import MistralForCausalLM

        return MistralForCausalLM

    def get_all_quant_overrides(self) -> dict:
        return dict(self.extra_quant_overrides)


BehemothR1_123BQkvA16Config = _BehemothR1_123BQkvA16Config(
    model_id="/media/fmodels/TheDrummer/Behemoth-R1-123B-v2",
    trust_remote_code=False,
    streaming=False,
    base_quant_cfg="NVFP4_DEFAULT_CFG",
    extra_quant_overrides={
        # The whole point: NVFP4 weights, BF16 activations, on q/k/v only.
        # Weight quantizers stay enabled. o_proj is untouched and stays W4A4.
        "*q_proj*input_quantizer": {"enable": False},
        "*k_proj*input_quantizer": {"enable": False},
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
    },
)
