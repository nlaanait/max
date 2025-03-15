# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Literal, Optional, Sequence, cast

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.graph.weights import Weights, WeightsAdapter
from max.nn import Module, Signals
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextContext,
)
from max.pipelines.dataprocessing import batch_padded_tokens_and_mask
from max.pipelines.interfaces import LogProbabilities
from max.pipelines.kv_cache import (
    KVCacheInputs,
    KVCacheManager,
    KVCacheParams,
    estimate_kv_cache_size,
    load_kv_manager,
)
from max.pipelines.log_probabilities import compute_log_probabilities
from transformers import AutoConfig

from .distributed_llama import DistributedLlama3
from .llama3 import Llama3
from .model_config import Llama3Config
from .naive_llama3 import NaiveLlama3

logger = logging.getLogger("max.pipelines")


class Llama3Inputs(ModelInputs):
    """A class representing inputs for the Llama3 model.

    This class encapsulates the input tensors required for the Llama3 model
    execution.
    """

    tokens: np.ndarray | Tensor
    """Tensor containing the input token IDs."""

    input_row_offsets_or_attn_mask: np.ndarray | Tensor
    """Tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    def __init__(
        self,
        tokens: np.ndarray | Tensor,
        input_row_offsets_or_attn_mask: np.ndarray | Tensor,
        signal_buffers: list[Tensor],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        """
        Args:
            tokens: Input token IDs.
            input_row_offsets_or_attn_mask: Input row offsets (ragged tensors)
                or attention mask (padded tensors).
            signal_buffers: Device buffers used for synchronization in
                communication collectives.
        """
        self.tokens = tokens
        self.input_row_offsets_or_attn_mask = input_row_offsets_or_attn_mask
        self.signal_buffers = signal_buffers
        self.kv_cache_inputs = kv_cache_inputs

    @property
    def input_row_offsets(self) -> np.ndarray | Tensor:
        """Gets the row offsets of the ragged input sequence."""
        # TODO(bduke): this should implement a ragged tensor interface.
        return self.input_row_offsets_or_attn_mask


class LlamaModelBase(PipelineModel[TextContext]):
    """Base Llama pipeline model implementation."""

    model: Model
    """Compiled and initialized model ready for inference."""

    signal_buffers: list[Tensor]
    """Device buffers used for synchronization in communication collectives."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"]
    """Normalization layer."""

    logits_postprocessor: Callable[[TensorValue], TensorValue] | None = None
    """Postprocessor for the logits."""

    state_dict: dict[str, Any]
    """Weights to load into the model."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> None:
        """
        Args:
            pipeline_config: The configuration for this pipeline.
            session: The container for the runtime for this model.
        """
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
        )
        self.model = self.load_model(session)

        # Initialize state needed for communication collectives.
        self.signal_buffers = (
            [
                Tensor.zeros(
                    shape=(Signals.NUM_BYTES,),
                    dtype=DType.uint8,
                    device=dev,
                )
                for dev in self.devices
            ]
            if len(self.devices) > 1
            # Skip creating buffers for single-device, where communication
            # collectives shouldn't be called.
            else []
        )

    # TODO(zheng): These get_kv_params calls in PipelineModel(s) should probably be
    # a config interface / method.
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Llama3Config.get_kv_params(
            huggingface_config,
            n_devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        model_inputs = cast(Llama3Inputs, model_inputs)
        curr_kv_cache_inputs = model_inputs.kv_cache_inputs or ()
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets_or_attn_mask,
            *model_inputs.signal_buffers,
            *curr_kv_cache_inputs,
            copy_inputs_to_device=(
                not self.kv_cache_config.cache_strategy.uses_opaque()
            ),
        )

        if self.pipeline_config.enable_echo:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0]),
                logits=cast(Tensor, model_outputs[1]),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(Tensor, model_outputs[0])
            )

    def _prepare_ragged_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        # Get input_row_offsets: start and end position of each batch in the
        # combined total_seq_len dimension.
        input_row_offsets = np.cumsum(
            [0] + [ctx.active_length for ctx in context_batch],
            dtype=np.uint32,
        )

        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])

        return Llama3Inputs(
            tokens=Tensor.from_numpy(tokens).to(self.devices[0]),
            input_row_offsets_or_attn_mask=Tensor.from_numpy(
                input_row_offsets
            ).to(self.devices[0]),
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def _prepare_padded_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        # Get tokens and seq_ids
        tokens = [ctx.next_tokens for ctx in context_batch]

        # Pad tokens and compute attention mask for the batch.
        max_seq_len = self.kv_manager.max_sequence_length
        start_pos = [max_seq_len] * len(context_batch)
        next_tokens_batch, _, attn_mask = batch_padded_tokens_and_mask(
            start_pos=start_pos,
            tokens=tokens,
            pad_to_multiple_of=self.pipeline_config.pad_to_multiple_of,
        )

        return Llama3Inputs(
            tokens=next_tokens_batch,
            input_row_offsets_or_attn_mask=attn_mask,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Llama3Inputs:
        """Prepare the inputs for the first pass in multistep execution."""
        if self.kv_cache_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_initial_token_inputs(
                context_batch, kv_cache_inputs
            )
        else:
            return self._prepare_padded_initial_token_inputs(
                context_batch, kv_cache_inputs
            )

    def _prepare_ragged_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: Llama3Inputs,
    ) -> Llama3Inputs:
        row_offsets_size = (
            prev_model_inputs.input_row_offsets_or_attn_mask.shape[0]
        )
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        return Llama3Inputs(
            tokens=next_tokens,
            input_row_offsets_or_attn_mask=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Llama3Inputs:
        """Prepare the inputs for the next token in multistep execution.
        This should avoid any device synchronization or copy operations.
        """
        prev_model_inputs = cast(Llama3Inputs, prev_model_inputs)
        if self.kv_cache_config.cache_strategy.uses_opaque():
            return self._prepare_ragged_next_token_inputs(
                next_tokens, prev_model_inputs
            )
        else:
            # TODO(MODELS-407): Consider deleting the padded path entirely.
            msg = "multistep unsupported for padded token batches"
            raise ValueError(msg)

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    def load_kv_manager(
        self,
        session: InferenceSession,
        available_cache_memory: int,
    ) -> KVCacheManager:
        return load_kv_manager(
            params=self.get_kv_params(
                huggingface_config=self.huggingface_config,
                n_devices=len(self.devices),
                kv_cache_config=self.kv_cache_config,
                cache_dtype=self.encoding.cache_dtype,
            ),
            max_batch_size=self.pipeline_config.max_batch_size,
            max_seq_len=self.calculate_max_seq_len(
                self.pipeline_config, huggingface_config=self.huggingface_config
            ),
            num_layers=self.get_num_layers(
                huggingface_config=self.huggingface_config
            ),
            devices=self.devices,
            available_cache_memory=available_cache_memory,
            page_size=self.kv_cache_config.kv_cache_page_size,
            session=session,
        )

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: List[Device],
        huggingface_config: AutoConfig,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> int:
        """Estimates the size of the kv cache in bytes."""
        return estimate_kv_cache_size(
            params=cls.get_kv_params(
                huggingface_config=huggingface_config,
                n_devices=len(devices),
                kv_cache_config=kv_cache_config,
                cache_dtype=cache_dtype,
            ),
            max_batch_size=pipeline_config.max_batch_size,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config,
                huggingface_config=huggingface_config,
            ),
            num_layers=cls.get_num_layers(
                huggingface_config=huggingface_config,
            ),
            available_cache_memory=available_cache_memory,
            devices=devices,
        )

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        # Pre-allocate a buffer for input_row_offsets in multistep execution.
        # We do this to avoid materializing and copying a buffer with each multistep step
        assert self.pipeline_config.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Tensor.from_numpy(
            np.arange(self.pipeline_config.max_batch_size + 1, dtype=np.uint32)
        ).to(self.devices[0])

        logger.info("Building and compiling model...")
        before = time.perf_counter()
        graph = self._build_graph(self.weights, self.adapter)
        model = session.load(graph, weights_registry=self.state_dict)
        after = time.perf_counter()
        logger.info(
            f"Building and compiling model took {after - before:.6f} seconds"
        )

        return model

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[TensorValue]
    ) -> List[tuple[TensorValue, ...]]:
        kv_params = self.get_kv_params(
            huggingface_config=self.huggingface_config,
            n_devices=len(self.devices),
            kv_cache_config=self.kv_cache_config,
            cache_dtype=self.encoding.cache_dtype,
        )
        n_devices = kv_params.n_devices
        fetch_types = self.kv_manager.input_symbols()[0]
        len_of_kv_tuple_per_dev = len(list(fetch_types))
        kv_caches_per_dev = [
            tuple(
                kv_inputs_flat[
                    i * len_of_kv_tuple_per_dev : (i + 1)
                    * len_of_kv_tuple_per_dev
                ]
            )
            for i in range(n_devices)
        ]
        return kv_caches_per_dev

    def _build_opaque_graph(
        self, weights: Weights, adapter: Optional[WeightsAdapter] = None
    ) -> Graph:
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        # NOTE: input_row_offsets_len should be batch_size + 1.
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        huggingface_config = self.huggingface_config
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        model_config = Llama3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=self.logits_postprocessor,
            norm_method=self.norm_method,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
        )
        nn_model: Module
        if len(self.devices) > 1:
            kv_cache_args = self.kv_manager.input_symbols()
            flattened_kv_types = [
                kv_type for sublist in kv_cache_args for kv_type in sublist
            ]

            # Create metadata for signal buffers.
            signals = Signals(
                devices=(DeviceRef(d.label, d.id) for d in self.devices)
            )

            nn_model = DistributedLlama3(model_config)

            # Load weights. We allow the weight types to be overriden due to
            # multiple quantization encodings in GGUF checkpoints.
            nn_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
            )
            self.state_dict = nn_model.state_dict()
            with Graph(
                getattr(self.huggingface_config, "model_type", "llama3"),
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *signals.input_types(),
                    *flattened_kv_types,
                ],
            ) as graph:
                tokens, input_row_offsets, *variadic_args = graph.inputs

                # Multi-GPU passes a signal buffer per device: unmarshal those.
                signal_buffers = [
                    v.buffer for v in variadic_args[: len(self.devices)]
                ]

                # Unmarshal the remaining arguments, which are for KV cache.
                kv_cache = [
                    v.tensor for v in variadic_args[len(self.devices) :]
                ]

                kv_caches_per_dev = self._unflatten_kv_inputs(kv_cache)

                outputs = nn_model(
                    tokens.tensor,
                    signal_buffers,
                    kv_caches_per_dev,
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph
        else:
            nn_model = Llama3(model_config)

            # Load weights. We allow the weight types to be overriden due to
            # multiple quantization encodings in GGUF checkpoints.
            nn_model.load_state_dict(
                state_dict,
                override_quantization_encoding=True,
                weight_alignment=1,
            )
            self.state_dict = nn_model.state_dict()
            with Graph(
                "llama3",
                input_types=[
                    tokens_type,
                    input_row_offsets_type,
                    *self.kv_manager.input_symbols()[0],
                ],
            ) as graph:
                tokens, input_row_offsets, *kv_cache_inputs = graph.inputs
                outputs = nn_model(
                    tokens.tensor,
                    [inp.tensor for inp in kv_cache_inputs],
                    input_row_offsets=input_row_offsets,
                )
                graph.output(*outputs)
                return graph

    def _build_graph(
        self, weights: Weights, adapter: Optional[WeightsAdapter] = None
    ) -> Graph:
        if self.kv_cache_config.cache_strategy.uses_opaque():
            return self._build_opaque_graph(weights, adapter)

        tokens_type = TensorType(DType.int64, shape=["batch_size", "seq_len"])
        attn_mask_type = TensorType(
            DType.float32, shape=["batch_size", "seq_len", "post_seq_len"]
        )

        if len(self.devices) > 1:
            raise ValueError(
                "Naive mode does not support distributed execution"
            )

        kv_inputs = self.kv_manager.input_symbols()[0]
        if adapter:
            state_dict = adapter(
                dict(weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            state_dict = {key: value.data() for key, value in weights.items()}
        model_config = Llama3Config.generate(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            dtype=self.dtype,
            n_devices=len(self.devices),
            logits_postprocessor=self.logits_postprocessor,
            norm_method=self.norm_method,
            cache_dtype=self.encoding.cache_dtype,
            kv_cache_config=self.kv_cache_config,
        )
        nn_model = NaiveLlama3(model_config)

        # Load weights. We allow the weight types to be overriden due to
        # multiple quantization encodings in GGUF checkpoints.
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
        )
        self.state_dict = nn_model.state_dict()

        with Graph(
            getattr(self.huggingface_config, "model_type", "llama3"),
            input_types=[
                tokens_type,
                attn_mask_type,
                *kv_inputs,
            ],
        ) as graph:
            tokens, attention_mask, k_cache, v_cache, start_pos, _ = (
                graph.inputs
            )
            mask_dtype = (
                self.dtype
                if self.pipeline_config.model_config.quantization_encoding
                in [
                    SupportedEncoding.float32,
                    SupportedEncoding.bfloat16,
                ]
                else (
                    DType.float32
                    if self.devices[0].label == "cpu"
                    else DType.bfloat16
                )
            )
            logits = nn_model(
                tokens.tensor,
                attention_mask.tensor.cast(mask_dtype),
                k_cache.buffer,
                v_cache.buffer,
                start_pos.tensor,
            )[0]

            if self.pipeline_config.enable_echo:
                graph.output(logits[:, -1], logits)
            else:
                graph.output(logits[:, -1])

            return graph

    def compute_log_probabilities(
        self,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Tensor,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None] | None:
        if any(echo for echo in batch_echo):
            if model_outputs.logits is None:
                logger.warning(
                    "Could not get logprobs with echo because the full logits"
                    f" were not returned by {self.pipeline_config.model_config.model_path}"
                    " model. Please ensure that this model is started with "
                    "`--enable-echo`."
                )
                assert not self.pipeline_config.enable_echo, (
                    "Echo was enabled but logits were not returned."
                )
                return None
            logits = model_outputs.logits.to_numpy()

        llama3_inputs = cast(Llama3Inputs, model_inputs)
        next_token_logits = cast(
            Tensor, model_outputs.next_token_logits
        ).to_numpy()

        sampled_tokens = next_tokens.to_numpy()
        if self.kv_cache_config.cache_strategy.uses_opaque():
            # Handle the ragged inputs
            tokens = cast(Tensor, llama3_inputs.tokens).to_numpy()
            input_row_offsets = cast(
                Tensor, llama3_inputs.input_row_offsets
            ).to_numpy()

            def _get_logits_and_samples(
                batch_index: int, echo: bool
            ) -> tuple[np.ndarray, np.ndarray]:
                if echo:
                    start_offset = input_row_offsets[batch_index]
                    end_offset = input_row_offsets[batch_index + 1]
                    batch_logits = logits[start_offset:end_offset]
                    samples = np.concatenate(
                        (
                            tokens[start_offset + 1 : end_offset],
                            sampled_tokens[batch_index : batch_index + 1],
                        )
                    )
                else:
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        else:
            # Handle batched inputs. Llama pads them to the right so the seq
            # lengths can be computed by finding the first 0 token.
            tokens = cast(np.ndarray, llama3_inputs.tokens)
            seq_lens = np.sum(tokens > 0, axis=1)

            def _get_logits_and_samples(
                batch_index: int, echo: bool
            ) -> tuple[np.ndarray, np.ndarray]:
                if echo:
                    seq_len = seq_lens[batch_index]
                    padded_tokens = tokens[batch_index]

                    batch_logits = logits[batch_index, :seq_len, :]
                    samples = np.concatenate(
                        (
                            padded_tokens[1:seq_len],
                            sampled_tokens[batch_index : batch_index + 1],
                        )
                    )
                else:
                    batch_logits = next_token_logits[
                        batch_index : batch_index + 1, :
                    ]
                    samples = sampled_tokens[batch_index : batch_index + 1]
                return batch_logits, samples

        return compute_log_probabilities(
            _get_logits_and_samples, batch_top_n, batch_echo
        )


class Llama3Model(LlamaModelBase):
    """Llama 3 pipeline model implementation."""

    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    """Normalization layer."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: Optional[WeightsAdapter] = None,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
        )
