# Copyright (c) Kyutai, all rights reserved.
# MLX TTS module for moshi-server - uses moshi_mlx Python package

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import time
import typing as tp

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
from moshi_mlx import models
from moshi_mlx.models.generate import LmGen
from moshi_mlx.modules.conditioner import (
    ConditionAttributes,
    ConditionTensor,
    dropout_all_conditions,
)
from moshi_mlx.utils.sampling import Sampler
from moshi_mlx.models.tts import (
    Entry,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    script_to_entries,
)
from moshi_mlx.utils.loaders import hf_get
import json


class MaskFlags(Enum):
    HAS_PCM = 1
    IS_EOS = 2
    WORD_FINISHED = 4
    AR_STEP = 8
    MISSING_WORDS = 16


def flags_out_from_mask_(flags_out: np.ndarray, mask: np.ndarray, value: int):
    flags_out[mask] |= value


@dataclass
class Config:
    hf_repo: str = DEFAULT_DSM_TTS_REPO
    voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO
    voice_folder: str = str(Path.home() / 'models/tts-voices')
    default_voice: str = "barack_demo.wav"

    temp: float = 0.6
    cfg_coef: float = 2.0
    quantize: int | None = 8  # 8-bit quantization by default

    max_padding: int = 8
    initial_padding: int = 2
    final_padding: int = 4
    padding_between: int = 1
    padding_bonus: float = 0.0


def init(batch_size: int, config_override: dict) -> 'TTSService':
    print(f"[MLX] Initializing with batch_size={batch_size}", file=sys.stderr, flush=True)
    print(f"[MLX] config_override: {config_override}", file=sys.stderr, flush=True)

    config = Config(**{k: v for k, v in config_override.items() if hasattr(Config, k)})

    print(f"[MLX] Loading model from {config.hf_repo}", file=sys.stderr, flush=True)

    # Load config from HF repo
    raw_config_path = hf_get("config.json", config.hf_repo)
    with open(raw_config_path, "r") as f:
        raw_config = json.load(f)

    # Load model weights
    mimi_weights = hf_get(raw_config["mimi_name"], config.hf_repo)
    moshi_name = raw_config.get("moshi_name", "model.safetensors")
    moshi_weights = hf_get(moshi_name, config.hf_repo)
    tokenizer_path = hf_get(raw_config["tokenizer_name"], config.hf_repo)

    # Build LM model
    lm_config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    print(f"[MLX] Loading model weights from {moshi_weights}", file=sys.stderr, flush=True)
    model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    # Apply quantization
    if config.quantize is not None:
        print(f"[MLX] Quantizing model to {config.quantize} bits", file=sys.stderr, flush=True)
        nn.quantize(model.depformer, bits=config.quantize)
        for layer in model.transformer.layers:
            nn.quantize(layer.self_attn, bits=config.quantize)
            nn.quantize(layer.gating, bits=config.quantize)

    # Load tokenizers
    print(f"[MLX] Loading text tokenizer from {tokenizer_path}", file=sys.stderr, flush=True)
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    print(f"[MLX] Loading audio tokenizer from {mimi_weights}", file=sys.stderr, flush=True)
    generated_codebooks = lm_config.generated_codebooks
    audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
    audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

    # Create TTS model
    cfg_coef_conditioning = None
    tts_model = TTSModel(
        model,
        audio_tokenizer,
        text_tokenizer,
        voice_repo=config.voice_repo,
        temp=config.temp,
        cfg_coef=config.cfg_coef,
        max_padding=config.max_padding,
        initial_padding=config.initial_padding,
        final_padding=config.final_padding,
        padding_bonus=config.padding_bonus,
        raw_config=raw_config,
    )

    if tts_model.valid_cfg_conditionings:
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0

    # Load voices
    print(f"[MLX] Loading voices from {config.voice_folder}", file=sys.stderr, flush=True)
    voice_suffix = tts_model.voice_suffix
    all_attributes = {}
    voice_folder = Path(config.voice_folder)

    if tts_model.multi_speaker:
        for file in voice_folder.glob(f'**/*{voice_suffix}'):
            relative = file.relative_to(voice_folder)
            name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
            name_normalized = name.replace('\\', '/')
            try:
                attributes = tts_model.make_condition_attributes([file, file], cfg_coef=cfg_coef_conditioning)
            except Exception as e:
                print(f"[MLX WARNING] Failed to load voice {name_normalized}: {e}", file=sys.stderr)
            else:
                all_attributes[name] = attributes
                if name != name_normalized:
                    all_attributes[name_normalized] = attributes

        if not all_attributes:
            raise RuntimeError(
                f"No voices found in {voice_folder}/**/*{voice_suffix}"
            )

        if config.default_voice not in all_attributes:
            print(f"[MLX WARNING] Default voice {config.default_voice} not found, using first available", file=sys.stderr)
            config.default_voice = list(all_attributes.keys())[0]

    service = TTSService(
        batch_size=batch_size,
        default_attribute_name=config.default_voice,
        all_attributes=all_attributes,
        tts_model=tts_model,
        cfg_coef_conditioning=cfg_coef_conditioning,
        padding_between=config.padding_between,
    )

    print("[MLX] Ready to roll!", file=sys.stderr, flush=True)
    return service


def _make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    return dropout_all_conditions(all_attributes)


@dataclass
class TTSGen:
    """MLX TTS generator - handles streaming generation for a single client."""
    tts_model: TTSModel
    attributes: tp.Sequence[ConditionAttributes]
    on_frame: tp.Optional[tp.Callable[[mx.array], None]] = None

    def __post_init__(self):
        tts_model = self.tts_model
        attributes = self.attributes

        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        self.ct = None
        self.cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if self.ct is None:
                    self.ct = _ct
                else:
                    self.ct = ConditionTensor(self.ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if self.cross_attention_src is None:
                    self.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        def _on_audio_hook(audio_tokens):
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=tts_model.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            cfg_coef=tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )
        self.lm_gen.step(
            input_tokens, ct=self.ct, cross_attention_src=self.cross_attention_src
        )
        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1
        if frame is not None:
            if self.on_frame is not None:
                self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


@dataclass
class ClientState:
    is_complete: bool = False
    gen: TTSGen | None = None
    pcm_queue: list = field(default_factory=list)
    word_finished: bool = False

    def reset(self, tts_model: TTSModel, attributes: tp.Sequence[ConditionAttributes]):
        self.is_complete = False
        self.pcm_queue = []
        self.word_finished = False

        def on_frame(frame):
            if (frame == -1).any():
                return
            pcm = tts_model.mimi.decode_step(frame[:, :, None])
            pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
            self.pcm_queue.append(pcm)

        self.gen = TTSGen(tts_model, attributes, on_frame=on_frame)


@dataclass
class TTSService:
    batch_size: int
    default_attribute_name: str
    all_attributes: dict[str, tp.Sequence[ConditionAttributes]]
    tts_model: TTSModel
    cfg_coef_conditioning: float | None = None
    padding_between: int = 1

    clients: list[ClientState] = field(default_factory=list)

    def __post_init__(self):
        for _ in range(self.batch_size):
            self.clients.append(ClientState())

        # Warm up
        print("[MLX] Warming up...", file=sys.stderr, flush=True)
        mx.eval(self.tts_model.mimi.parameters())
        mx.eval(self.tts_model.lm.parameters())
        print("[MLX] Warmup complete", file=sys.stderr, flush=True)

    def _get_attributes(self, voice: str | None) -> tp.Sequence[ConditionAttributes]:
        if voice and voice in self.all_attributes:
            return self.all_attributes[voice]
        return self.all_attributes[self.default_attribute_name]

    def step(self, updates: list[tuple[int, list[int], str | None, int | None]],
             pcm_out: np.ndarray, flags_out: np.ndarray, code_out: np.ndarray) -> None:
        """Process one step for all active clients.

        Args:
            updates: List of (batch_idx, tokens, voice, seed) tuples
            pcm_out: Output array for PCM data [batch_size, 1920]
            flags_out: Output array for flags [batch_size]
            code_out: Output array for codes [batch_size, 33]
        """
        machine = self.tts_model.machine
        pad = machine.token_ids.pad

        flags_out[:] = 0

        # Process updates
        for b, new_entry, voice, seed in updates:
            client = self.clients[b]

            if new_entry[0] == -1:
                # Reset - new client
                if seed is not None:
                    mx.random.seed(seed)
                attributes = self._get_attributes(voice)
                client.reset(self.tts_model, [attributes])
                new_entry = new_entry[1:]

            if client.gen is None:
                continue

            if new_entry == [-2]:
                # End of stream
                client.is_complete = True
            elif new_entry[0] == pad:
                # Padding entry
                padding = len(new_entry)
                client.gen.append_entry(Entry([], '', padding=padding))
            elif new_entry:
                # Text tokens
                padding = 0
                if self.padding_between > 0:
                    padding = max(0, self.padding_between + len(new_entry) - 1)
                client.gen.append_entry(Entry(new_entry, '', padding=padding))

        # Process each client
        for b, client in enumerate(self.clients):
            if client.gen is None:
                continue

            gen = client.gen
            state = gen.state
            lookahead = self.tts_model.machine.second_stream_ahead

            # Check if we can run
            can_run = False
            if client.is_complete:
                can_run = True
            elif state.entries:
                if lookahead == 0:
                    can_run = True
                elif not state.entries[0].tokens:
                    can_run = True
                else:
                    remaining = lookahead + 1
                    for entry in state.entries:
                        if entry.tokens:
                            remaining -= 1
                        if remaining <= 0:
                            can_run = True
                            break

            if not can_run:
                flags_out[b] |= MaskFlags.MISSING_WORDS.value
                continue

            flags_out[b] |= MaskFlags.AR_STEP.value

            # Run one step
            if client.is_complete and (len(state.entries) == 0 and state.end_step is not None):
                # Check if we're done
                real_end = (
                    state.end_step + self.tts_model.delay_steps +
                    self.tts_model.final_padding + max(self.tts_model.lm.delays)
                )
                if gen.offset >= real_end:
                    flags_out[b] |= MaskFlags.IS_EOS.value
                    client.gen = None
                    continue

            # Step the generator
            if client.is_complete:
                if len(state.entries) > 0 or state.end_step is not None:
                    gen._step()
                else:
                    gen._step()  # Continue for delay
            else:
                gen.process()

            # Check for PCM output
            if client.pcm_queue:
                pcm = client.pcm_queue.pop(0)
                pcm_out[b, :len(pcm)] = pcm
                flags_out[b] |= MaskFlags.HAS_PCM.value
