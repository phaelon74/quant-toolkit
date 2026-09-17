from .behemoth_r1_123b import BehemothR1_123BConfig
from .behemoth_r1_123b_q import BehemothR1_123BQConfig
from .behemoth_r1_123b_qkv import BehemothR1_123BQkvConfig
from .behemoth_r1_123b_qkv_a16 import BehemothR1_123BQkvA16Config
from .gemma4_31b import Gemma4_31BConfig
from .glm5 import Glm5Config
from .glm5_1 import Glm51Config
from .minimax_m25 import MinimaxM25Config
from .minimax_m27 import MinimaxM27Config
from .qwen3_5_122b import Qwen35_122BConfig
from .qwen3_5_moe import Qwen35MoeConfig
from .qwen3_5_moe_noshared import Qwen35MoeNoSharedConfig

_CONFIGS = {
    "behemoth_r1_123b": BehemothR1_123BConfig,
    "behemoth_r1_123b_q": BehemothR1_123BQConfig,
    "behemoth_r1_123b_qkv": BehemothR1_123BQkvConfig,
    "behemoth_r1_123b_qkv_a16": BehemothR1_123BQkvA16Config,
    "gemma4_31b": Gemma4_31BConfig,
    "glm5": Glm5Config,
    "glm5_1": Glm51Config,
    "minimax_m25": MinimaxM25Config,
    "minimax_m27": MinimaxM27Config,
    "qwen3_5_122b": Qwen35_122BConfig,
    "qwen3_5_moe": Qwen35MoeConfig,
    "qwen3_5_moe_noshared": Qwen35MoeNoSharedConfig,
}

AVAILABLE_MODELS = list(_CONFIGS.keys())


def load_config(name: str):
    if name not in _CONFIGS:
        raise ValueError(f"Unknown model config: {name}. Available: {AVAILABLE_MODELS}")
    return _CONFIGS[name]
