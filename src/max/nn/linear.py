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

"""Multi-layer Perceptron."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import Weights

from .clamp import clamp
from .comm import Allreduce
from .kernels import swish_glu
from .layer import Layer, Module


class LinearV2(Module):
    """
    Applies a linear transformation to incoming data: :math:`y = xW^T + b`.

    This layer implements a fully connected layer where inputs are multiplied
    by a weight matrix and optionally added with a bias vector.
    Both weights and bias initially reside on CPU, and the model init phase
    moves them to :obj:`device`.

    Example:

    .. code-block:: python

        linear_layer = LinearV2(
            in_dim=256,
            out_dim=128,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="linear",
            has_bias=True
        )

        # Input tensor of shape: [batch, ..., 256]
        input_tensor: TensorValue
        output = linear_layer(input_tensor)
    """

    weight: Weight
    """The weight matrix stored on CPU with shape (out_dim, in_dim).
    Model init transposes the weight and moves it to :obj:`device`."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_dim,).
    Model init moves the bias to :obj:`device` if present."""

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device
        self.clip_weight = clip_weight

        self.weight = Weight(
            name=f"{name}.weight" if name else "weight",
            dtype=dtype,
            shape=(out_dim, in_dim),
            device=DeviceRef.CPU() if self.device else None,
            quantization_encoding=quantization_encoding,
        )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_dim,),
                device=DeviceRef.CPU() if self.device else None,
                quantization_encoding=quantization_encoding,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        weight: TensorValue = self.weight
        if self.device:
            weight = weight.to(self.device)

        if self.clip_weight:
            weight = clamp(weight, -self.clip_weight, self.clip_weight)

        if self.weight.quantization_encoding:
            res = ops.qmatmul(
                self.weight.quantization_encoding,
                None,
                x,
                weight,
            )
        else:
            res = x @ weight.T
        if self.bias is not None:
            bias = self.bias.to(self.device) if self.device else self.bias
            res += bias
        return res


def _allocate_if_needed(value: Weights | Weight, dtype, shape) -> Weight:
    if isinstance(value, Weight):
        return value
    else:
        return value.weight.allocate(dtype, shape)


@dataclass
class Linear(Layer):
    """A unified linear layer that delegates to either regular or quantized implementation."""

    weight: TensorValueLike
    bias: TensorValueLike | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        weight = TensorValue(self.weight)
        res = x @ weight.T
        if self.bias is not None:
            res += self.bias
        return res

    @classmethod
    def create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None = None,
        quantization_config: QuantizationConfig | None = None,
    ) -> "Linear":
        """Factory method to create a Linear layer with appropriate implementation."""
        if not quantization_encoding:
            weight = _allocate_if_needed(
                weights, dtype, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return Linear(weight=weight, bias=bias_weight)
        else:
            return QLinear._create(
                dtype,
                quantization_encoding,
                in_features,
                out_features,
                weights,
                bias,
                quantization_config,
            )


@dataclass
class QLinear(Linear):
    """A quantized fully connected layer."""

    # Because Linear.bias is optional and Linear is a dataclass and we inherit from Linear, all our fields must be optional even if it doesn't make logical sense
    quantization_encoding: QuantizationEncoding | None = None

    @classmethod
    def _create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None,
        quantization_config: QuantizationConfig | None,
    ) -> "Linear":
        if quantization_encoding != QuantizationEncoding.GPTQ:
            weight = _allocate_if_needed(
                weights, dtype, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return QLinear(
                weight=weight,
                bias=bias_weight,
                # GGUF weights can have different quantization per weight
                quantization_encoding=weight.quantization_encoding,
            )
        else:
            return GPTQLinear._create(
                dtype,
                quantization_encoding,
                in_features,
                out_features,
                weights,
                bias,
                quantization_config,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.quantization_encoding is not None
        weight = TensorValue(self.weight)
        res = ops.qmatmul(
            self.quantization_encoding,
            None,
            x,
            weight,
        )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


@dataclass
class GPTQLinear(QLinear):
    "A Linear layer for GPTQ encoding"

    # Because QLinear has optional fields, so must we, since we subclass QLinear
    quantization_config: QuantizationConfig | None = None
    perm_idx: TensorValueLike | None = None

    @classmethod
    def _create(
        cls,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        in_features: int,
        out_features: int,
        weights: Weights | Weight,
        bias: Weights | Weight | None,
        quantization_config: QuantizationConfig | None,
    ) -> "Linear":
        """Internal method to create a Linear layer from GPTQ weights."""

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )

        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        desc_act = quantization_config.desc_act

        perm_idx = None

        if isinstance(weights, Weights) and weights.qweight.exists():
            orig_quantized_weights = [weights.qweight, weights.scales]
            quantized_weights = []
            for idx, qw in enumerate(orig_quantized_weights):
                orig = qw.allocate()
                # TODO(AITLIB-135): allocate_as_bytes is only available for
                # safetensors. This isn't a problem right now because gptq is
                # only present for safetensors
                weight_bytes = qw.allocate_as_bytes()  # type: ignore
                assert len(orig.shape) == 2
                reshaped = ops.reshape(
                    weight_bytes,
                    (orig.shape[0] * orig.dtype.size_in_bytes, orig.shape[1]),
                ).transpose(0, 1)
                quantized_weights.append(reshaped)

            weight = ops.concat(
                (quantized_weights[0], quantized_weights[1]), axis=1
            ).transpose(0, 1)

            if desc_act:
                perm_idx = weights.g_idx.allocate(
                    DType.int32,
                    [out_features],
                )
                # hack: argsort the perm_idx array
                weights._allocated[perm_idx.name] = np.argsort(  # type: ignore
                    weights._allocated[perm_idx.name]  # type: ignore
                ).astype(np.int32)

            return GPTQLinear(
                weight=weight,
                bias=None,
                quantization_encoding=quantization_encoding,
                quantization_config=quantization_config,
                perm_idx=perm_idx,
            )

        else:
            weight = _allocate_if_needed(
                weights, DType.bfloat16, [in_features, out_features]
            )
            bias_weight = (
                _allocate_if_needed(bias, dtype, [out_features])
                if bias
                else None
            )
            return Linear(weight, bias_weight)

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.quantization_encoding is not None
        weight = TensorValue(self.weight)
        if self.perm_idx is not None:
            perm_idx = TensorValue(self.perm_idx)
            res = ops.qmatmul(
                self.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.quantization_encoding,
                self.quantization_config,
                x,
                weight,
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


@dataclass
class GPTQLinearV2(LinearV2):
    "A Linear layer for GPTQ encoding"

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quantization_config: QuantizationConfig | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias with
        GPTQ quantization.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The data type for both weights and bias.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            has_bias: When :obj:`True`, adds a bias vector to the layer.
                Defaults to :obj:`False`.
            quantization_encoding: The quantization encoding of the weights.
            quantization_config: Extra config for the weight quantization.
        """
        del out_dim, dtype  # Unused.
        if has_bias:
            raise ValueError("has_bias=True is not supported in GPTQLinear.")

        # Skip LinearV2 initialization.
        Module.__init__(self)
        self.device = device
        self.qweight = Weight(
            name="qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=DeviceRef.CPU() if device else None,
            quantization_encoding=quantization_encoding,
        )
        self.scales = Weight(
            name="scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=DeviceRef.CPU() if device else None,
            quantization_encoding=quantization_encoding,
        )

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )
        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        self.quantization_config = quantization_config

        desc_act = self.quantization_config.desc_act
        self.perm_idx = None
        if desc_act:
            self.perm_idx = Weight(
                "perm_idx",
                DType.int32,
                [in_dim],
                device=DeviceRef.CPU() if device else None,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        assert self.qweight.quantization_encoding is not None
        qweight_dtype, qweight_shape = self.qweight.original_dtype_and_shape
        qweight = ops.reshape(
            self.qweight,
            (qweight_shape[0] * qweight_dtype.size_in_bytes, qweight_shape[1]),
        ).transpose(0, 1)

        scales_dtype, scales_shape = self.scales.original_dtype_and_shape
        scales = ops.reshape(
            self.scales,
            (scales_shape[0] * scales_dtype.size_in_bytes, scales_shape[1]),
        ).transpose(0, 1)
        weight = ops.concat((qweight, scales), axis=1).transpose(0, 1)
        if self.device:
            weight = weight.to(self.device)
        if self.perm_idx is not None:
            perm_idx: TensorValue = self.perm_idx
            if self.device:
                perm_idx = perm_idx.to(self.device)
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                x,
                weight,
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


@dataclass
class MLP(Layer):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    gate_proj: Linear
    down_proj: Linear
    up_proj: Linear

    def __call__(self, x: TensorValueLike) -> TensorValue:
        if (
            self.gate_proj.bias is None
            and self.up_proj.bias is None
            and TensorValue(x).rank == 2
            and TensorValue(x).device is not None
            and TensorValue(x).device != DeviceRef.CPU()
            and False  # GEX-1476: This causes elaboration errors - disable swish_glu pathway.
        ):
            return self.down_proj(
                swish_glu(
                    x,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                )
            )

        return self.down_proj(ops.silu(self.gate_proj(x)) * self.up_proj(x))  # type: ignore


class MLPV2(Module):
    """
    Simple multi-layer perceptron composed of three linear layers.
    Uses SiLU activation function.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        linear_cls: Callable[..., LinearV2] = LinearV2,
        devices: Sequence[DeviceRef] = (),
    ):
        """
        Args:
            dtype: DType to use for the layer weights, which should match the
                input dtype.
            quantization_encoding: Quantization encoding of the layer weights.
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of dimension used to project the inputs.
            linear_cls: Linear class to use to create the projection layers.
            devices: Devices to run the `MLP` layer. If multiple are provided,
                the first device is used instead. Use `DistributedMLP` to use
                all devices.
        """
        super().__init__()
        self.devices = devices
        self.gate_proj = linear_cls(  # [ffl, hidden]
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=devices[0] if devices else None,
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0] if devices else None,
            quantization_encoding=quantization_encoding,
        )
        self.up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=devices[0] if devices else None,
            quantization_encoding=quantization_encoding,
        )
        self.quantization_encoding = quantization_encoding

    def __call__(self, x: TensorValueLike) -> TensorValue:
        if (
            self.gate_proj.bias is None
            and self.up_proj.bias is None
            and TensorValue(x).rank == 2
            and TensorValue(x).device is not None
            and TensorValue(x).device != DeviceRef.CPU()
            and False  # GEX-1476: This causes elaboration errors - disable swish_glu pathway.
        ):
            return self.down_proj(
                swish_glu(
                    x,
                    self.gate_proj.weight,
                    self.up_proj.weight,
                )
            )
        if self.quantization_encoding:
            return self.down_proj(
                ops.silu(self.gate_proj(x)) * self.up_proj(x)  # type: ignore
            )
        else:
            # Optimization to compute a single matmul by merging the
            # gate and up projection weights.
            feed_forward_length = self.gate_proj.weight.shape[0]
            gate_proj_weight: TensorValue = self.gate_proj.weight
            if self.gate_proj.device:
                gate_proj_weight = gate_proj_weight.to(self.gate_proj.device)
            up_proj_weight: TensorValue = self.up_proj.weight
            if self.up_proj.device:
                up_proj_weight = up_proj_weight.to(self.up_proj.device)
            output = x @ ops.concat((gate_proj_weight, up_proj_weight)).T
            gate_out, up_out = ops.split(
                output, [feed_forward_length, feed_forward_length], axis=1
            )
            return self.down_proj(ops.silu(gate_out) * up_out)


class DistributedMLP(MLPV2):
    """A distributed multi-layer perceptron.

    This class has the same state keys as the non-distributed MLP Layer.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_devices = len(self.devices)

        def col_sharding_strategy(weight: Weight, i) -> TensorValue:
            col_size = int(weight.shape[1]) // self.num_devices
            return weight[:, i * col_size : (i + 1) * col_size]

        def row_sharding_strategy(weight: Weight, i) -> TensorValue:
            row_size = int(weight.shape[0]) // self.num_devices
            return weight[i * row_size : (i + 1) * row_size, :]

        self.gate_proj.weight.set_sharding_strategy(row_sharding_strategy)
        self.down_proj.weight.set_sharding_strategy(col_sharding_strategy)
        self.up_proj.weight.set_sharding_strategy(row_sharding_strategy)

        # Create normal MLP layers for each device. These layers and weights are
        # not recorded by the nn.Module and do not appear in the state dict.
        self.list_of_mlps = []
        for n, device in enumerate(self.devices):
            layer = MLPV2(*args, **kwargs)

            layer.gate_proj.device = device
            layer.gate_proj.weight = self.gate_proj.weight.shard(n, device)

            layer.down_proj.device = device
            layer.down_proj.weight = self.down_proj.weight.shard(n, device)

            layer.up_proj.device = device
            layer.up_proj.weight = self.up_proj.weight.shard(n, device)

            self.list_of_mlps.append(layer)

        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def __call__(  # type: ignore[override]
        self, x: list[TensorValue], signal_buffers: list[BufferValue]
    ) -> list[TensorValue]:
        """Applies a linear transformation to the input data.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on :obj:`device`.
            signal_buffers: Buffers for peer-to-peer communication in allreduce.

        Returns:
            Output tensor of shape ``(..., out_dim)``.
            The result resides on the device specified in :obj:`device`.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        mlp_outs = [self.list_of_mlps[i](x[i]) for i in range(self.num_devices)]
        return self.allreduce(mlp_outs, signal_buffers)
