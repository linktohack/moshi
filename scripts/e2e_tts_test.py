#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "websockets",
#     "numpy",
#     "scipy",
#     "msgpack",
# ]
# ///
# Streaming playback: use --play to hear audio in real-time via ffplay/mpv
"""
End-to-End TTS Test Script

This script connects to a TTS server via WebSocket, sends text,
receives audio, and saves it as a WAV file for analysis.

Usage:
    uv run scripts/e2e_tts_test.py --port 8089 --text "Hello world" --output python_output.wav

    # Stream audio in real-time while receiving:
    uv run scripts/e2e_tts_test.py --port 8089 --text "Hello world" --play

    # Read text from stdin:
    echo "Hello world" | uv run scripts/e2e_tts_test.py --port 8089 --play
    cat story.txt | uv run scripts/e2e_tts_test.py --port 8089 --play
"""

import argparse
import asyncio
import io
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import websockets
import msgpack


def find_player() -> tuple[str, list[str]] | None:
    """Find an available audio player that can stream raw PCM from stdin."""
    # mpv: excellent player, handles streaming raw audio well
    if shutil.which("mpv"):
        return ("mpv", [
            "mpv",
            "--no-video",
            "--demuxer=rawaudio",
            "--demuxer-rawaudio-rate=24000",
            "--demuxer-rawaudio-channels=1",
            "--demuxer-rawaudio-format=floatle",
            "--no-cache",
            "-"
        ])
    # ffplay: part of ffmpeg
    if shutil.which("ffplay"):
        return ("ffplay", [
            "ffplay", "-f", "f32le", "-ar", "24000", "-ac", "1",
            "-nodisp", "-autoexit", "-"
        ])
    return None


async def generate_tts(
    host: str,
    port: int,
    text: str,
    voice: str,
    output_path: str,
    temperature: float = 0.6,
    top_k: int = 250,
    seed: int = 299792458,
    format: str = "PcmMessagePack",
    timeout: float = 120.0,
    debug: bool = False,
    debug_output_dir: str | None = None,
    play: bool = False,
):
    """Connect to TTS server and generate audio.

    If play=True, streams audio to ffplay/mpv for real-time playback.
    """

    # Build query parameters
    params = {
        "voice": voice,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "format": format,
        "auth_id": "public_token",  # Required for authorization
    }

    # Add debug flag if enabled
    if debug:
        params["debug"] = "true"
        if debug_output_dir:
            params["debug_output_dir"] = debug_output_dir

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    uri = f"ws://{host}:{port}/api/tts_streaming?{query_string}"

    print(f"Connecting to {uri}")

    pcm_chunks = []
    words_with_timestamps = []

    # Timing measurements
    import time
    start_time = time.perf_counter()
    first_audio_time = None

    # Set up streaming player if requested
    player_proc = None
    if play:
        player_info = find_player()
        if player_info:
            player_name, player_cmd = player_info
            print(f"Streaming audio via {player_name}...")
            player_proc = subprocess.Popen(
                player_cmd,
                stdin=subprocess.PIPE,
            )
        else:
            print("Warning: No suitable player found (ffplay, mpv). Install ffmpeg or mpv for streaming playback.")

    try:
        async with websockets.connect(uri, max_size=10_000_000) as ws:
            connect_time = time.perf_counter() - start_time
            print(f"Connected in {connect_time*1000:.1f}ms! Sending text: '{text}'")

            # Send text
            await ws.send(text)

            # Signal end of text
            await ws.send(b"\0")

            print("Receiving audio...")

            # Receive messages until connection closes
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        # Try to decode as msgpack
                        try:
                            decoded = msgpack.unpackb(message, raw=False)
                            msg_type = decoded.get("type")

                            if msg_type == "Audio":
                                pcm = decoded.get("pcm", [])
                                if pcm:
                                    if first_audio_time is None:
                                        first_audio_time = time.perf_counter() - start_time
                                        print(f"  *** First audio at {first_audio_time*1000:.1f}ms (TTFA) ***")
                                    pcm_array = np.array(pcm, dtype=np.float32)
                                    pcm_chunks.append(pcm_array)
                                    print(f"  Received audio chunk: {len(pcm)} samples")
                                    # Stream to player
                                    if player_proc and player_proc.stdin:
                                        try:
                                            player_proc.stdin.write(pcm_array.tobytes())
                                            player_proc.stdin.flush()
                                        except BrokenPipeError:
                                            pass  # Player closed

                            elif msg_type == "Text":
                                text_content = decoded.get("text", "")
                                start_s = decoded.get("start_s", 0)
                                stop_s = decoded.get("stop_s", 0)
                                words_with_timestamps.append({
                                    "text": text_content,
                                    "start_s": start_s,
                                    "stop_s": stop_s,
                                })
                                print(f"  Word: '{text_content}' ({start_s:.2f}s - {stop_s:.2f}s)")

                            elif msg_type == "OggOpus":
                                # OggOpus format - we requested pcm_msgpack so this shouldn't happen
                                print(f"  Received OggOpus data (unexpected)")

                            elif msg_type == "Ready":
                                print("  Server ready")

                            elif msg_type == "Error":
                                error_msg = decoded.get("message", "Unknown error")
                                print(f"  Error: {error_msg}")
                                break

                            else:
                                print(f"  Unknown message type: {msg_type}")

                        except msgpack.UnpackException:
                            # Raw PCM data
                            pcm_array = np.frombuffer(message, dtype=np.float32)
                            pcm_chunks.append(pcm_array)
                            print(f"  Received raw PCM: {len(pcm_array)} samples")
                            # Stream to player
                            if player_proc and player_proc.stdin:
                                try:
                                    player_proc.stdin.write(pcm_array.tobytes())
                                    player_proc.stdin.flush()
                                except BrokenPipeError:
                                    pass  # Player closed

                    else:
                        print(f"  Text message: {message[:100]}...")

            except websockets.exceptions.ConnectionClosed as e:
                print(f"Connection closed: {e}")

    except asyncio.TimeoutError:
        print(f"Timeout after {timeout}s")
        if player_proc:
            player_proc.stdin.close()
            player_proc.wait()
        return None, None

    except Exception as e:
        print(f"Error: {e}")
        if player_proc:
            player_proc.stdin.close()
            player_proc.wait()
        return None, None

    # Close player stdin to signal end of stream, then wait for it to finish
    if player_proc:
        try:
            player_proc.stdin.close()
            player_proc.wait(timeout=30)  # Wait for playback to finish
        except Exception:
            player_proc.kill()

    if not pcm_chunks:
        print("No audio received!")
        return None, None

    # Concatenate all PCM chunks
    total_time = time.perf_counter() - start_time
    pcm = np.concatenate(pcm_chunks)
    audio_duration = len(pcm) / 24000
    generation_time = total_time - connect_time  # exclude connection overhead
    rtf = audio_duration / generation_time if generation_time > 0 else 0

    print(f"\nTotal audio: {len(pcm)} samples ({audio_duration:.2f}s at 24kHz)")
    print(f"Generation time: {generation_time*1000:.0f}ms (wall clock: {total_time*1000:.0f}ms)")
    print(f"Real-time factor: {rtf:.2f}x (>{1:.0f}x means faster than real-time)")

    # Save as WAV
    save_wav(output_path, pcm, sample_rate=24000)
    print(f"Saved to: {output_path}")

    return pcm, words_with_timestamps


def save_wav(path: str, pcm: np.ndarray, sample_rate: int = 24000):
    """Save PCM data as WAV file."""
    # Clip to [-1, 1] and convert to int16
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm_int16 = (pcm * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_int16.tobytes())


def analyze_audio(pcm: np.ndarray, sample_rate: int = 24000):
    """Analyze audio characteristics."""
    print("\n=== Audio Analysis ===")

    duration = len(pcm) / sample_rate
    print(f"Duration: {duration:.2f}s")

    # Energy statistics
    rms = np.sqrt(np.mean(pcm ** 2))
    peak = np.max(np.abs(pcm))
    print(f"RMS energy: {rms:.4f}")
    print(f"Peak amplitude: {peak:.4f}")

    # Check for silence
    silence_threshold = 0.01
    silent_samples = np.sum(np.abs(pcm) < silence_threshold)
    silence_ratio = silent_samples / len(pcm)
    print(f"Silence ratio: {silence_ratio:.2%}")

    # Check for clipping
    clip_threshold = 0.99
    clipped_samples = np.sum(np.abs(pcm) > clip_threshold)
    clip_ratio = clipped_samples / len(pcm)
    print(f"Clipping ratio: {clip_ratio:.4%}")

    # Zero crossings (rough measure of frequency content)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(pcm))) > 0)
    zc_rate = zero_crossings / duration
    print(f"Zero-crossing rate: {zc_rate:.0f} Hz")

    # Simple spectral analysis
    try:
        from scipy import signal

        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(pcm, fs=sample_rate, nperseg=1024)

        # Find dominant frequencies
        mean_spectrum = np.mean(Sxx, axis=1)
        dominant_freqs = f[np.argsort(mean_spectrum)[-5:]]
        print(f"Dominant frequencies: {dominant_freqs.astype(int)} Hz")

        # Energy in speech band (300-3000 Hz)
        speech_band = (f >= 300) & (f <= 3000)
        speech_energy = np.sum(mean_spectrum[speech_band])
        total_energy = np.sum(mean_spectrum)
        speech_ratio = speech_energy / total_energy if total_energy > 0 else 0
        print(f"Speech band energy ratio: {speech_ratio:.2%}")

    except ImportError:
        print("(scipy not available for spectral analysis)")

    return {
        "duration": duration,
        "rms": rms,
        "peak": peak,
        "silence_ratio": silence_ratio,
        "clip_ratio": clip_ratio,
        "zc_rate": zc_rate,
    }


def compare_audio(pcm1: np.ndarray, pcm2: np.ndarray, name1: str, name2: str):
    """Compare two audio samples."""
    print(f"\n=== Comparing {name1} vs {name2} ===")

    # Duration comparison
    dur1 = len(pcm1) / 24000
    dur2 = len(pcm2) / 24000
    print(f"Duration: {name1}={dur1:.2f}s, {name2}={dur2:.2f}s, diff={abs(dur1-dur2):.2f}s")

    # Energy comparison
    rms1 = np.sqrt(np.mean(pcm1 ** 2))
    rms2 = np.sqrt(np.mean(pcm2 ** 2))
    print(f"RMS: {name1}={rms1:.4f}, {name2}={rms2:.4f}, ratio={rms1/rms2:.2f}")

    # Correlation (align to shorter length)
    min_len = min(len(pcm1), len(pcm2))
    if min_len > 0:
        corr = np.corrcoef(pcm1[:min_len], pcm2[:min_len])[0, 1]
        print(f"Correlation: {corr:.4f}")

    # MSE
    mse = np.mean((pcm1[:min_len] - pcm2[:min_len]) ** 2)
    print(f"MSE: {mse:.6f}")


async def main():
    parser = argparse.ArgumentParser(description="E2E TTS Test")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8089, help="TTS server port")
    parser.add_argument("--text", type=str, default=None,
                       help="Text to synthesize. If not provided, reads from stdin.")
    parser.add_argument("--voice", type=str,
                       default="unmute-prod-website/ex04_narration_longform_00001.wav")
    parser.add_argument("--output", type=str, default="tts_output.wav")
    parser.add_argument("--output-dir", type=str, default="./tts_test_output")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=299792458)
    parser.add_argument("--analyze-only", type=str, help="Analyze existing WAV file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (server dumps tensors)")
    parser.add_argument("--debug-output-dir", type=str, help="Directory for debug tensor dumps")
    parser.add_argument("--play", action="store_true", help="Stream audio to player (ffplay/mpv) for real-time playback")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get text from args or stdin
    text = args.text
    if text is None:
        if sys.stdin.isatty():
            # No piped input and no --text provided, use default
            text = "Hello, this is a test of the text to speech system. The quick brown fox jumps over the lazy dog."
        else:
            # Read from stdin
            text = sys.stdin.read().strip()
            if not text:
                print("Error: No text provided via --text or stdin")
                sys.exit(1)

    if args.analyze_only:
        # Just analyze an existing file
        import wave
        with wave.open(args.analyze_only, 'rb') as wav:
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            pcm = pcm.astype(np.float32) / 32767
            sample_rate = wav.getframerate()
        analyze_audio(pcm, sample_rate)
        return

    output_path = output_dir / args.output

    print(f"=== E2E TTS Test ===")
    print(f"Server: {args.host}:{args.port}")
    print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"Voice: {args.voice}")
    print(f"Output: {output_path}")
    print()

    pcm, words = await generate_tts(
        host=args.host,
        port=args.port,
        text=text,
        voice=args.voice,
        output_path=str(output_path),
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        debug=args.debug,
        debug_output_dir=args.debug_output_dir,
        play=args.play,
    )

    if pcm is not None:
        analyze_audio(pcm)

        # Print transcript
        if words:
            print("\n=== Transcript ===")
            for w in words:
                print(f"  [{w['start_s']:.2f}s - {w['stop_s']:.2f}s] {w['text']}")


if __name__ == "__main__":
    asyncio.run(main())
