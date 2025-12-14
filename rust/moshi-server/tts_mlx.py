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

    # Load voices - only look for .safetensors files directly
    print(f"[MLX] Loading voices from {config.voice_folder}", file=sys.stderr, flush=True)
    voice_suffix = tts_model.voice_suffix
    all_attributes = {}
    voice_folder = Path(config.voice_folder)

    if tts_model.multi_speaker:
        # Only look for files that end with the voice suffix (e.g., .1e68beda@240.safetensors)
        for file in voice_folder.glob(f'**/*{voice_suffix}'):
            relative = file.relative_to(voice_folder)
            name = str(relative.with_name(relative.name.removesuffix(voice_suffix)))
            name_normalized = name.replace('\\', '/')
            try:
                attributes = tts_model.make_condition_attributes([file, file], cfg_coef=cfg_coef_conditioning)
                all_attributes[name] = attributes
                if name != name_normalized:
                    all_attributes[name_normalized] = attributes
            except Exception as e:
                # Only warn for files that should work (actual safetensors files)
                print(f"[MLX WARNING] Failed to load voice {name_normalized}: {e}", file=sys.stderr)

        if not all_attributes:
            raise RuntimeError(
                f"No voices found in {voice_folder}/**/*{voice_suffix}"
            )

        print(f"[MLX] Loaded {len(all_attributes)} voices", file=sys.stderr, flush=True)

        if config.default_voice not in all_attributes:
            # Try adding .wav suffix
            default_with_wav = config.default_voice + ".wav" if not config.default_voice.endswith(".wav") else config.default_voice
            default_without_wav = config.default_voice.removesuffix(".wav")
            if default_with_wav in all_attributes:
                config.default_voice = default_with_wav
            elif default_without_wav in all_attributes:
                config.default_voice = default_without_wav
            else:
                print(f"[MLX WARNING] Default voice {config.default_voice} not found, using first available", file=sys.stderr)
                config.default_voice = list(all_attributes.keys())[0]
        print(f"[MLX] Using default voice: {config.default_voice}", file=sys.stderr, flush=True)

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
class ClientState:
    """State for a single TTS client."""
    is_complete: bool = False
    offset: int = 0
    state: tp.Any = None  # State machine state
    lm_gen: tp.Any = None
    ct: tp.Any = None
    cross_attention_src: tp.Any = None
    pcm_queue: list = field(default_factory=list)
    word_consumed_this_step: bool = False
    tts_model: tp.Any = None


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
        print(f"[MLX] second_stream_ahead={self.tts_model.machine.second_stream_ahead}", file=sys.stderr, flush=True)
        print(f"[MLX] delay_steps={self.tts_model.delay_steps}", file=sys.stderr, flush=True)
        print("[MLX] Warmup complete", file=sys.stderr, flush=True)

    def _get_attributes(self, voice: str | None) -> tp.Sequence[ConditionAttributes]:
        if voice:
            # Try exact match
            if voice in self.all_attributes:
                return self.all_attributes[voice]
            # Try with/without .wav suffix
            voice_with_wav = voice if voice.endswith(".wav") else voice + ".wav"
            voice_without_wav = voice.removesuffix(".wav")
            if voice_with_wav in self.all_attributes:
                return self.all_attributes[voice_with_wav]
            if voice_without_wav in self.all_attributes:
                return self.all_attributes[voice_without_wav]
        return self.all_attributes[self.default_attribute_name]

    def _reset_client(self, client: ClientState, voice: str | None, seed: int | None):
        """Reset a client for new generation."""
        if seed is not None:
            mx.random.seed(seed)

        client.is_complete = False
        client.offset = 0
        client.pcm_queue = []
        client.word_consumed_this_step = False
        client.tts_model = self.tts_model
        client.state = self.tts_model.machine.new_state([])

        attributes = [self._get_attributes(voice)]

        if self.tts_model.cfg_coef != 1.0:
            if self.tts_model.valid_cfg_conditionings:
                raise ValueError("Model trained with CFG distillation")
            nulled = _make_null(attributes)
            attributes = attributes + nulled

        assert self.tts_model.lm.condition_provider is not None
        client.ct = None
        client.cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = self.tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if client.ct is None:
                    client.ct = _ct
                else:
                    client.ct = ConditionTensor(client.ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = self.tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if client.cross_attention_src is None:
                    client.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        # Create text hook that tracks word consumption
        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                # Handle both scalar and list tokens (MLX may return nested structure)
                while isinstance(token, list):
                    token = token[0] if token else 0

                out_token, consumed_new_word = self.tts_model.machine.process(client.offset, client.state, token)
                if consumed_new_word:
                    client.word_consumed_this_step = True
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        def _on_audio_hook(audio_tokens):
            delays = self.tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if client.offset < delay + self.tts_model.delay_steps:
                    audio_tokens[q] = self.tts_model.machine.token_ids.zero

        client.lm_gen = LmGen(
            self.tts_model.lm,
            max_steps=self.tts_model.max_gen_length,
            text_sampler=Sampler(temp=self.tts_model.temp),
            audio_sampler=Sampler(temp=self.tts_model.temp),
            cfg_coef=self.tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
        )

    def _client_step(self, client: ClientState) -> tp.Optional[np.ndarray]:
        """Run a single step for a client. Returns PCM if available."""
        if client.lm_gen is None:
            return None

        client.word_consumed_this_step = False

        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )
        client.lm_gen.step(
            input_tokens, ct=client.ct, cross_attention_src=client.cross_attention_src
        )
        frame = client.lm_gen.last_audio_tokens()
        client.offset += 1

        # Debug: log state machine status periodically
        if client.offset % 20 == 0:
            entries_len = len(client.state.entries) if client.state else 0
            end_step = client.state.end_step if client.state else None
            print(f"[MLX DEBUG] offset={client.offset} entries={entries_len} end_step={end_step} complete={client.is_complete}", file=sys.stderr, flush=True)

        if frame is not None and not (frame == -1).any():
            pcm = self.tts_model.mimi.decode_step(frame[:, :, None])
            pcm = np.array(mx.clip(pcm[0, 0], -1, 1))
            return pcm
        return None

    def _is_client_active(self, client: ClientState) -> bool:
        """Check if client can run (has enough lookahead)."""
        if client.state is None or client.lm_gen is None:
            return False
        if client.is_complete:
            return True
        if not client.state.entries:
            return False

        lookahead = self.tts_model.machine.second_stream_ahead
        if lookahead == 0:
            return True
        if not client.state.entries[0].tokens:
            # Next entry is just padding
            return True

        remaining = lookahead + 1
        for entry in client.state.entries:
            if entry.tokens:
                remaining -= 1
            if remaining <= 0:
                return True
        return False

    def _is_client_done(self, client: ClientState) -> bool:
        """Check if client generation is complete."""
        if not client.is_complete:
            return False
        if client.state is None:
            return True
        if len(client.state.entries) > 0 or client.state.end_step is None:
            return False
        real_end = (
            client.state.end_step + self.tts_model.delay_steps +
            self.tts_model.final_padding + max(self.tts_model.lm.delays)
        )
        # Debug first time we check done condition
        if client.offset == client.state.end_step + 1:
            print(f"[MLX DEBUG] end_step={client.state.end_step} delay_steps={self.tts_model.delay_steps} final_padding={self.tts_model.final_padding} max_delays={max(self.tts_model.lm.delays)} real_end={real_end}", file=sys.stderr, flush=True)
        return client.offset >= real_end

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

            if new_entry and new_entry[0] == -1:
                # Reset - new client
                self._reset_client(client, voice, seed)
                new_entry = new_entry[1:]

            if client.state is None:
                continue

            if new_entry == [-2]:
                # End of stream
                client.is_complete = True
            elif new_entry and new_entry[0] == pad:
                # Padding entry
                padding = len(new_entry)
                client.state.entries.append(Entry([], '', padding=padding))
            elif new_entry:
                # Text tokens
                padding = 0
                if self.padding_between > 0:
                    padding = max(0, self.padding_between + len(new_entry) - 1)
                client.state.entries.append(Entry(new_entry, '', padding=padding))

        # Process each client
        for b, client in enumerate(self.clients):
            if client.state is None:
                continue

            # Check if done
            if self._is_client_done(client):
                flags_out[b] |= MaskFlags.IS_EOS.value
                client.state = None
                client.lm_gen = None
                continue

            # Check if we can run
            if not self._is_client_active(client):
                flags_out[b] |= MaskFlags.MISSING_WORDS.value
                continue

            flags_out[b] |= MaskFlags.AR_STEP.value

            # Run one step
            pcm = self._client_step(client)

            # Check for word consumed
            if client.word_consumed_this_step:
                flags_out[b] |= MaskFlags.WORD_FINISHED.value

            # Check for PCM output
            if pcm is not None:
                pcm_out[b, :len(pcm)] = pcm
                flags_out[b] |= MaskFlags.HAS_PCM.value
