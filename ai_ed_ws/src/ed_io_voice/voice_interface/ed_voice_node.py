#!/usr/bin/env python3
# Offline voice I/O node for Jetson (Vosk ASR + WebRTC VAD + Piper TTS)
# - Filters garbage like "##########" so TTS only speaks real language
# - Forces mic by name/index to avoid "pulse/monitor" desktop audio
# - Uses Piper CLI (text via stdin) with aplay playback, plus pico2wave/espeak fallback
#
# Env knobs (all optional):
#   ED_MIC_DEVICE="Jabra"           # name hint (case-insensitive)
#   ED_MIC_INDEX=0                  # explicit PortAudio input index (overrides name hint)
#   ED_MIC_STRICT=1                 # fail if ED_MIC_DEVICE not found
#   ED_BLOCK_PULSE=1                # skip "pulse"/"monitor" devices
#   ED_ASR_RATE=16000               # Vosk small model sample rate
#   ED_VAD_LEVEL=2                  # 0..3 (3 most aggressive)
#   ED_VAD_SILENCE_MS=600           # end-of-utterance silence gap
#   ED_VAD_FRAME_MS=30              # 10/20/30 only
#   ED_MIN_CHARS=4                  # drop utterances shorter than this
#   ED_MIN_WORDS=1                  # require at least this many words
#   ED_VOSK_MODEL=~/.../vosk-model-small-en-us-0.15
#   ED_PIPER_BIN=piper              # path to piper CLI
#   ED_PIPER_MODEL=~/.../en_GB-cori-high.onnx
#   ED_PIPER_CONFIG=~/.../en_GB-cori-high.onnx.json
#   ALSA_PCM_BUFFER_TIME=250000     # playback stability (microseconds)
#
# ROS:
#   Publishes recognized text to /ed/speech_command (dofbot_pro_interface/SpeechCommand)
#   Provides /ed/speak (dofbot_pro_interface/Speak) for TTS

import os
import re
import json
import threading
import queue
import subprocess
import tempfile
import time
from typing import Optional

import rclpy
from rclpy.node import Node

from dofbot_pro_interface.msg import SpeechCommand
from dofbot_pro_interface.srv import Speak

# --- audio / vad / asr ---
import sounddevice as sd         # mic capture via PortAudio
import webrtcvad                 # robust voice activity detection
from vosk import Model, KaldiRecognizer  # offline ASR (Kaldi)

# ------------------ Utils ------------------

def shutil_which(exe: str) -> Optional[str]:
    try:
        from shutil import which
        return which(exe)
    except Exception:
        return None

def _looks_like_language(txt: str) -> bool:
    """
    Quick heuristic to avoid speaking classifier noise like '###########'
    or almost-all punctuation.
    """
    if not txt:
        return False
    t = txt.strip()
    if not t:
        return False
    # drop if mostly punctuation / hashes
    letters = sum(ch.isalpha() for ch in t)
    if letters < max(3, len(t) // 10):
        return False
    if t.count('#') > len(t) * 0.3:
        return False
    # avoid giant repeated characters
    if re.search(r'(.)\1{8,}', t):
        return False
    return True

def _chunk_text_for_tts(text: str, max_chars: int = 400):
    """
    Split text into chunks near sentence boundaries so Piper doesn’t get one huge blob.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) <= max_chars:
        return [text]
    # split by sentence-ish endings
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

def _device_is_blocked(name: str) -> bool:
    if not _bool_env("ED_BLOCK_PULSE", True):
        return False
    nl = name.lower()
    return ("pulse" in nl) or ("monitor" in nl)

def find_input_device_index(name_hint: Optional[str]) -> Optional[int]:
    """
    Return PortAudio input device index by name_hint (case-insensitive).
    Skip pulse/monitor if ED_BLOCK_PULSE=1 (default).
    Fallback: default input; then first input-capable device.
    """
    devices = sd.query_devices()

    # explicit override wins
    if os.getenv("ED_MIC_INDEX") is not None:
        try:
            idx = int(os.getenv("ED_MIC_INDEX"))
            if 0 <= idx < len(devices) and devices[idx].get("max_input_channels", 0) > 0:
                if not _device_is_blocked(str(devices[idx].get("name",""))):
                    return idx
        except Exception:
            pass  # fall through

    # name hint
    if name_hint:
        hint = name_hint.lower()
        for i, d in enumerate(devices):
            nm = str(d.get("name", ""))
            if d.get("max_input_channels", 0) > 0 and hint in nm.lower():
                if not _device_is_blocked(nm):
                    return i

    # default input
    try:
        default_idx = sd.default.device[0]
        if default_idx is not None and default_idx >= 0:
            nm = str(devices[default_idx].get("name", ""))
            if devices[default_idx].get("max_input_channels", 0) > 0:
                if not _device_is_blocked(nm):
                    return default_idx
    except Exception:
        pass

    # first input-capable device (not blocked)
    for i, d in enumerate(devices):
        nm = str(d.get("name", ""))
        if d.get("max_input_channels", 0) > 0 and not _device_is_blocked(nm):
            return i
    return None

def piper_tts(text: str, model_path: str, config_path: str,
              length_scale: str = "1.0",
              noise_scale: str = "0.667",
              noise_w: str = "0.8",
              sentence_silence: str = "0.2",
              piper_bin: Optional[str] = None,
              use_cuda: bool = False) -> str:
    """
    Generate speech with Piper and return temp WAV path.
    Feeds text on stdin. Requires piper CLI and onnx/json files.
    """
    piper_bin = piper_bin or os.getenv("ED_PIPER_BIN", "piper")
    if not shutil_which(piper_bin):
        raise RuntimeError(f"piper binary not found: {piper_bin}. Install it or set ED_PIPER_BIN.")

    wav_path = tempfile.mktemp(prefix="ed_tts_", suffix=".wav")
    cmd = [
        piper_bin,
        "--model", model_path,
        "--config", config_path,
        "--length_scale", length_scale,
        "--noise_scale", noise_scale,
        "--noise_w", noise_w,
        "--sentence_silence", sentence_silence,
        "--output_file", wav_path,
    ]
    if use_cuda:
        cmd.append("--cuda")

    # Piper reads text from stdin (one or more sentences)
    subprocess.run(cmd, input=text, text=True, check=True)
    return wav_path

def play_wav_alsa(wav_path: str) -> None:
    """Play WAV via ALSA (aplay)."""
    aplay = shutil_which("aplay")
    if not aplay:
        raise RuntimeError("aplay not found. Install alsa-utils.")
    subprocess.run([aplay, "-q", wav_path], check=True)

# ------------------ Node ------------------

class EdVoiceNode(Node):
    """
    Offline voice I/O node:
      - Mic -> WebRTC VAD -> Vosk (ASR)
      - TTS service via Piper -> aplay (with pico2wave/espeak fallback)
    Publishes recognized utterances to /ed/speech_command
    Provides /ed/speak (Speak.srv) to synthesize responses.
    """

    def __init__(self):
        super().__init__("ed_voice_client")

        # --- Config from env (with sane defaults) ---
        self.mic_name_hint = os.getenv("ED_MIC_DEVICE") or os.getenv("MIC_HINT") or "Jabra"
        self.mic_strict = _bool_env("ED_MIC_STRICT", False)

        self.sample_rate = _int_env("ED_ASR_RATE", 16000)     # Vosk model = 16k
        self.vad_aggressiveness = _int_env("ED_VAD_LEVEL", 2) # 0..3 (3=most aggressive)
        self.vad_silence_ms = _int_env("ED_VAD_SILENCE_MS", 600)
        self.vad_frame_ms = _int_env("ED_VAD_FRAME_MS", 30)   # 10/20/30 only
        self.min_chars = _int_env("ED_MIN_CHARS", 4)
        self.min_words = _int_env("ED_MIN_WORDS", 1)

        self.vosk_model_dir = os.getenv(
            "ED_VOSK_MODEL",
            os.path.expanduser("~/ai_ed_ws/src/ed_io_voice/voice_interface/models/vosk-model-small-en-us-0.15")
        )

        # Piper voice files (you’ve downloaded these already)
        self.piper_bin = os.getenv("ED_PIPER_BIN", "piper")
        self.piper_model = os.getenv(
            "ED_PIPER_MODEL",
            os.path.expanduser("~/Downloads/piper_voices/en_GB-cori-high.onnx")
        )
        self.piper_config = os.getenv(
            "ED_PIPER_CONFIG",
            os.path.expanduser("~/Downloads/piper_voices/en_GB-cori-high.onnx.json")
        )
        self.piper_cuda = _bool_env("ED_PIPER_CUDA", False)

        # --- ROS I/O ---
        self.command_pub = self.create_publisher(SpeechCommand, "/ed/speech_command", 10)
        self.tts_srv = self.create_service(Speak, "/ed/speak", self._srv_speak)
        self.get_logger().info("TTS service ready: /ed/speak")

        # --- Initialize ASR/VAD ---
        if not os.path.isdir(self.vosk_model_dir):
            raise RuntimeError(f"Vosk model dir not found: {self.vosk_model_dir}")
        self.get_logger().info(f"Loading Vosk model: {self.vosk_model_dir}")
        self.vosk_model = Model(self.vosk_model_dir)
        self.rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
        self.rec.SetWords(True)

        if self.vad_frame_ms not in (10, 20, 30):
            raise ValueError("ED_VAD_FRAME_MS must be 10, 20, or 30")
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)
        self.bytes_per_frame = int(self.sample_rate * (self.vad_frame_ms / 1000.0)) * 2  # 16-bit mono = 2 bytes

        # mic device
        self.mic_index = find_input_device_index(self.mic_name_hint)
        if self.mic_index is None and self.mic_strict:
            raise RuntimeError(f"No input matches ED_MIC_DEVICE='{self.mic_name_hint}' and strict mode is on.")
        self.get_logger().info(
            f"Mic device: {self.mic_index} (hint='{self.mic_name_hint}') @ {self.sample_rate}Hz"
        )

        # state
        self.audio_q: "queue.Queue[bytes]" = queue.Queue(maxsize=100)
        self.speech_active = False
        self.last_voice_time = 0.0

        # start audio stream
        self._start_stream()

        # processing thread
        self.worker = threading.Thread(target=self._listen_loop, daemon=True)
        self.worker.start()
        self.get_logger().info("🎙️  EdVoiceNode is live (Vosk+WebRTC VAD+Piper).")

    # -------- Mic / Stream --------
    def _on_audio(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warn(f"Audio status: {status}")
        try:
            self.audio_q.put_nowait(bytes(indata))
        except queue.Full:
            # drop if backed up (should be rare)
            pass

    def _start_stream(self):
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.bytes_per_frame,  # exactly one VAD frame per callback
            callback=self._on_audio,
            device=self.mic_index
        )
        self.stream.start()

    # -------- VAD + ASR loop --------
    def _listen_loop(self):
        """
        Reads frames from queue, does VAD gating, streams to Vosk,
        and publishes final transcripts after a silence tail.
        """
        silence_limit = self.vad_silence_ms / 1000.0
        while rclpy.ok():
            try:
                frame = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ensure exact frame size
            if len(frame) != self.bytes_per_frame:
                if len(frame) > self.bytes_per_frame:
                    frame = frame[:self.bytes_per_frame]
                else:
                    frame = frame + b"\x00" * (self.bytes_per_frame - len(frame))

            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                continue  # bad frame; skip

            now = time.monotonic()
            # feed every frame to Vosk (handles silence too)
            self.rec.AcceptWaveform(frame)

            if is_speech:
                if not self.speech_active:
                    self.speech_active = True
                self.last_voice_time = now
            else:
                # if we were in speech and now silence exceeds threshold -> finalize
                if self.speech_active and (now - self.last_voice_time) >= silence_limit:
                    self._finalize_utterance()
                    self.speech_active = False
                    # reset recognizer for next utterance
                    self.rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
                    self.rec.SetWords(True)

    def _finalize_utterance(self):
        try:
            res = self.rec.FinalResult()
            data = json.loads(res or "{}")
            text = (data.get("text") or "").strip()
            # Basic cleanup
            text = re.sub(r'\s+', ' ', text)
            # Drop very short/filler
            if not text:
                return
            if len(text) < self.min_chars and len(text.split()) < self.min_words:
                return

            self.get_logger().info(f'👤 You said: "{text}"')
            msg = SpeechCommand()
            msg.command = text
            self.command_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Finalize utterance failed: {e}")

    # -------- TTS service --------
    def _srv_speak(self, request, response):
        raw_text = request.text_to_speak or ""
        text = raw_text.strip()
        # Guard against garbage (eg. classifier hashes)
        if not _looks_like_language(text):
            self.get_logger().warn("TTS: dropped noisy text")
            response.success = False
            return response

        self.get_logger().info(f'🗣️  Speaking: "{text}"')

        # Try Piper in sentence-sized chunks
        chunks = _chunk_text_for_tts(text, max_chars=400)
        try:
            for chunk in chunks:
                wav = piper_tts(
                    chunk,
                    self.piper_model,
                    self.piper_config,
                    piper_bin=self.piper_bin,
                    use_cuda=self.piper_cuda
                )
                try:
                    play_wav_alsa(wav)
                finally:
                    try:
                        os.remove(wav)
                    except Exception:
                        pass
            response.success = True
            return response
        except Exception as e:
            self.get_logger().warn(f"Piper TTS failed ({e}), trying pico2wave/espeak fallback.")
            ok = self._fallback_tts(text)
            response.success = ok
            return response

    def _fallback_tts(self, text: str) -> bool:
        # pico2wave -> aplay
        if shutil_which("pico2wave") and shutil_which("aplay"):
            wav = tempfile.mktemp(prefix="ed_pico_", suffix=".wav")
            try:
                subprocess.run(["pico2wave", "-l", "en-US", "-w", wav, text], check=True)
                play_wav_alsa(wav)
                return True
            except Exception as e:
                self.get_logger().error(f"pico2wave failed: {e}")
            finally:
                try:
                    os.remove(wav)
                except Exception:
                    pass
        # espeak -> aplay
        if shutil_which("espeak") and shutil_which("aplay"):
            try:
                cmd = f'espeak -s 170 -p 30 -v en-us --stdout "{text}" | aplay -q'
                subprocess.run(cmd, shell=True, check=True)
                return True
            except Exception as e:
                self.get_logger().error(f"espeak failed: {e}")
        return False

# ------------------ Main ------------------

def main(args=None):
    # Raise ALSA buffer if needed for stability
    os.environ.setdefault("ALSA_PCM_BUFFER_TIME", "250000")  # 250ms

    rclpy.init(args=args)
    node = EdVoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down voice node...")
        try:
            node.stream.stop()
            node.stream.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
