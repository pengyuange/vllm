# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3-Omni model (thinker part) - Dense version."""

from collections.abc import Mapping
from typing import Literal, cast

import numpy as np
import torch
import torch.nn as nn
from transformers.models.qwen3_omni.configuration_qwen3_omni import (
    Qwen3OmniConfig,
    Qwen3OmniThinkerConfig,
)
from transformers.models.qwen3_omni.processing_qwen3_omni import (
    Qwen3OmniProcessor,
)
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2_audio import Qwen2AudioProcessingInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.processor import cached_processor_from_config

from .interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .qwen2_5_omni_thinker import Qwen2_5OmniThinkerDummyInputsBuilder
from .qwen2_5_vl import Qwen2_5_VLProcessingInfo
from .qwen3 import Qwen3ForCausalLM, Qwen3Model
from .qwen3_omni_moe_thinker import (
    ISO639_1_SUPPORTED_LANGS,
    Qwen3OmniMoeAudioEncoder as Qwen3OmniAudioEncoder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3Omni_VisionTransformer,
)
from .utils import (
    WeightsMapper,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen3LLMModel(Qwen3Model):
    """Qwen3 dense LLM model with DeepStack support for Qwen3-Omni."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.deepstack_multiscale_layer_start = 1

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = layer_idx + self.start_layer

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):
    """Qwen3 dense LLM for causal LM, using Qwen3LLMModel (with DeepStack)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Bypass Qwen3ForCausalLM.__init__ to substitute Qwen3LLMModel
        super(Qwen3ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3LLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


class Qwen3OmniThinkerProcessingInfo(
    Qwen2AudioProcessingInfo, Qwen2_5_VLProcessingInfo
):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3OmniConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3OmniProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3OmniProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


Qwen3OmniThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


class Qwen3OmniThinkerMultiModalProcessor(Qwen3OmniMoeThinkerMultiModalProcessor):
    """Multimodal processor for Qwen3-Omni dense model.

    Inherits all processing logic from the MoE processor; ``self.info``
    resolves to ``Qwen3OmniThinkerProcessingInfo``, which returns
    ``Qwen3OmniProcessor`` and ``Qwen3OmniConfig`` instead of their MoE
    equivalents.
    """


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniThinkerMultiModalProcessor,
    info=Qwen3OmniThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniThinkerDummyInputsBuilder,
)
class Qwen3OmniThinkerForConditionalGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration
):
    """Inference-only Qwen3-Omni thinker (dense variant).

    Inherits all multimodal processing, DeepStack, and MRoPE logic from
    the MoE thinker.  The only structural differences are:

    * Language model backbone: dense ``Qwen3LLMForCausalLM`` (based on
      ``Qwen3ForCausalLM``) instead of ``Qwen3MoeLLMForCausalLM``.
    * Config/processor classes: ``Qwen3OmniConfig`` /
      ``Qwen3OmniThinkerConfig`` / ``Qwen3OmniProcessor``.
    """

    supported_languages = ISO639_1_SUPPORTED_LANGS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Call nn.Module.__init__ directly to bypass the MoE parent setup.
        nn.Module.__init__(self)
        self.vllm_config = vllm_config  # needed for torch compile forward context
        thinker_config: Qwen3OmniThinkerConfig = (
            vllm_config.model_config.hf_config.thinker_config
        )
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3OmniAudioEncoder(
                thinker_config.audio_config,
                prefix=maybe_prefix(prefix, "audio_tower"),
            )

        self.use_deepstack = hasattr(
            thinker_config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(thinker_config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = thinker_config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3Omni_VisionTransformer(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            # Register buffer for DeepStack multi-scale visual features
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        thinker_config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3LLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    thinker_config.text_config,
                    architectures=["Qwen3ForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(
            model_config, processor_cls=Qwen3OmniProcessor
        )
        return SpeechToTextConfig(
            max_audio_clip_s=processor.feature_extractor.chunk_length,
            sample_rate=processor.feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Construct a transcription/translation prompt for Qwen3-Omni."""
        instruction = "Transcribe" if task_type == "transcribe" else "Translate"
        instruction += " this audio"

        if task_type == "translate" and to_language is None:
            to_language = "en"

        full_lang_name = cls.supported_languages.get(language, "")
        full_lang_name_to = cls.supported_languages.get(to_language, "")

        if task_type == "transcribe" and full_lang_name:
            instruction += f" into {full_lang_name}"
        elif task_type == "translate":
            if full_lang_name:
                instruction += f" from {full_lang_name}"
            if full_lang_name_to:
                instruction += f" into {full_lang_name_to}"

        instruction += "."

        if request_prompt:
            instruction += f" {request_prompt}"

        processor = cached_processor_from_config(
            model_config, processor_cls=Qwen3OmniProcessor
        )
        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        user_content = f"{audio_placeholder}{instruction}"

        messages = [{"role": "user", "content": user_content}]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        audio_data = (audio, stt_config.sample_rate)
        prompts_dict = {"multi_modal_data": {"audio": audio_data}, "prompt": prompt}
        return cast(PromptType, prompts_dict)
