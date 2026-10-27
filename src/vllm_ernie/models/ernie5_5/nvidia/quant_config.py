# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 loading of 32x32 power-of-two block FP8 ERNIE 5 checkpoints."""

import torch

from vllm.config import VllmConfig
from vllm.config.quantization import QuantSpec
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Ernie5_5MXFP8Config(Fp8Config):
    """Runs a 32x32 ``scale_fmt: ue8m0`` block-FP8 checkpoint as MXFP8.

    A power-of-two 32x32 block scale is exactly the e8m0 scale of each of its
    32 rows, so the checkpoint maps losslessly onto native 1x32 MXFP8 linears
    and experts on Blackwell. Other devices keep the fp32-scale block-FP8 path.
    """

    def __init__(self, base: Fp8Config) -> None:
        self.__dict__.update(base.__dict__)

    @classmethod
    def maybe_wrap(
        cls, quant_config: QuantizationConfig | None, hf_quant_cfg: dict | None
    ) -> QuantizationConfig | None:
        if (
            type(quant_config) is Fp8Config
            and quant_config.weight_block_size == [MXFP8_BLOCK_SIZE] * 2
            and (hf_quant_cfg or {}).get("scale_fmt") == "ue8m0"
            and current_platform.is_device_capability_family(100)
        ):
            return cls(quant_config)
        return quant_config

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.quantization.modelopt import (
            CkptCtx,
            ModelOptLinearMethod,
            ModelOptMxFp8Config,
            ModelOptMxFp8FusedMoE,
        )

        if not isinstance(layer, LinearBase | RoutedExperts):
            return super().get_quant_method(layer, prefix)
        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
            match_mode=self.ignored_layers_match_mode,
        ):
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return UnquantizedFusedMoEMethod(layer.moe_config)
        if isinstance(layer, LinearBase):
            return ModelOptLinearMethod(
                QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic), CkptCtx()
            )
        return ModelOptMxFp8FusedMoE(
            ModelOptMxFp8Config(True, None, self.ignored_layers),
            layer.moe_config,
        )


def to_mxfp8_weight_scale(scale: torch.Tensor) -> torch.Tensor:
    """Convert power-of-two ``[N/32, K/32]`` fp32 scales to ``[N, K/32]`` e8m0."""
    bits = scale.float().view(torch.int32)
    if (bits & ~0x7F800000).any():
        raise ValueError("MXFP8 loading needs positive power-of-two scales.")
    return (bits >> 23).to(torch.uint8).repeat_interleave(MXFP8_BLOCK_SIZE, dim=0)


def use_ernie5_5_mxfp8(vllm_config: VllmConfig) -> bool:
    """Swap in `Ernie5_5MXFP8Config` when it applies; returns whether it did."""
    hf_quant_cfg = getattr(
        vllm_config.model_config.hf_config, "quantization_config", None
    )
    vllm_config.quant_config = Ernie5_5MXFP8Config.maybe_wrap(
        vllm_config.quant_config, hf_quant_cfg
    )
    if not isinstance(vllm_config.quant_config, Ernie5_5MXFP8Config):
        return False
    logger.info_once("Loading the 32x32 ue8m0 block-FP8 checkpoint as MXFP8.")
    return True
