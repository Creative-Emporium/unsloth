from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_DEEPSEEKV3_REGISTERED = False
_IS_DEEPSEEKR1_REGISTERED = False
_IS_DEEPSEEKR1_ZERO_REGISTERED = False
_IS_DEEPSEEKR1_DISTILL_REGISTERED = False
class DeepseekV3ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-V{version}"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

class DeepseekR1ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}" if version else base_name
        if size:
            key = f"{key}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key
    
# Deepseek V3 Model Meta
DeepseekV3Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16],
)

DeepseekV3_0324Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3-0324",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)

DeepseekR1Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16, QuantType.GGUF],
)

DeepseekR1ZeroMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="Zero",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)

DeepseekR1DistillMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1-Distill",
    instruct_tags=[None],
    model_version="Llama",
    model_sizes=["8", "70"],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types={"8": [QuantType.UNSLOTH, QuantType.GGUF], "70": [QuantType.GGUF]},
)

        # "Qwen-7B-unsloth-bnb-4bit",
        # "Qwen-1.5B-unsloth-bnb-4bit",
        # "Qwen-32B-GGUF",
        # "Llama-8B-GGUF",
        # "Qwen-14B-GGUF",
        # "Qwen-32B-bnb-4bit",
        # "Qwen-1.5B-GGUF",
        # "Qwen-14B-unsloth-bnb-4bit",
        # "Llama-70B-GGUF"

def register_deepseek_v3_models(include_original_model: bool = False):
    global _IS_DEEPSEEKV3_REGISTERED
    if _IS_DEEPSEEKV3_REGISTERED:
        return
    _register_models(DeepseekV3Meta, include_original_model=include_original_model)
    _register_models(DeepseekV3_0324Meta, include_original_model=include_original_model)
    _IS_DEEPSEEKV3_REGISTERED = True


def register_deepseek_r1_models(include_original_model: bool = False):
    global _IS_DEEPSEEKR1_REGISTERED
    if _IS_DEEPSEEKR1_REGISTERED:
        return
    _register_models(DeepseekR1Meta, include_original_model=include_original_model)
    _register_models(DeepseekR1ZeroMeta, include_original_model=include_original_model)
    _register_models(DeepseekR1DistillMeta, include_original_model=include_original_model)
    _IS_DEEPSEEKR1_REGISTERED = True

#register_deepseek_v3_models(include_original_model=True)
register_deepseek_r1_models(include_original_model=True)

def _list_deepseek_r1_distill_models():
    from unsloth.utils.hf_hub import ModelInfo as HfModelInfo
    from unsloth.utils.hf_hub import list_models
    models: list[HfModelInfo] = list_models(author="unsloth", search="Distill")
    for model in models:
        model_id = model.id
        model_name = model_id.split("/")[-1]
        # parse out only the version
        version = model_name.removeprefix("DeepSeek-R1-Distill-")
        print(version)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
