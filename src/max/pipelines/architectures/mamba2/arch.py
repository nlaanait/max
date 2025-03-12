from max.pipelines import PipelineTask, SupportedArchitecture, SupportedEncoding, TextTokenizer, WeightsFormat
from max.pipelines.kv_cache import KVCacheStrategy

from .model import Mamba2Model 

mamba_arch = SupportedArchitecture(
    name="MambaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["state-spaces/mamba-2.8b-hf",
    "state-spaces/mamba-1.4b-hf", 
    ],
    supported_encodings={SupportedEncoding.bfloat16: [KVCacheStrategy.NAIVE]},
    default_encoding=SupportedEncoding.bfloat16,
    pipeline_model=Mamba2Model,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors
)