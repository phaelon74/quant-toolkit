from .base import ModelQuantConfig


class _Gemma4_31BConfig(ModelQuantConfig):
    def get_model_cls(self):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration


Gemma4_31BConfig = _Gemma4_31BConfig(
    model_id="google/gemma-4-31B-it",
    trust_remote_code=False,
    streaming=False,
    extra_quant_overrides={
        "*vision_tower*": {"enable": False},
        "*vision_model*": {"enable": False},
        "*embed_tokens*weight_quantizer": {"enable": False},
        "*embed_tokens*input_quantizer": {"enable": False},
        "*lm_head*weight_quantizer": {"enable": False},
        "*lm_head*input_quantizer": {"enable": False},
        "*per_layer*weight_quantizer": {"enable": False},
        "*per_layer*input_quantizer": {"enable": False},
    },
)
