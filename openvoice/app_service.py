import os
from dataclasses import dataclass
from typing import Optional

import langid
import torch

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter


SUPPORTED_LANGUAGES = ["zh", "en"]
ENGLISH_STYLES = [
    "default",
    "whispering",
    "shouting",
    "excited",
    "cheerful",
    "terrified",
    "angry",
    "sad",
    "friendly",
]


@dataclass
class SynthesisRequest:
    prompt: str
    style: str
    audio_file_path: str
    agree: bool


@dataclass
class SynthesisResponse:
    success: bool
    info: str
    output_audio_path: Optional[str] = None
    reference_audio_path: Optional[str] = None


class OpenVoiceService:
    def __init__(self, checkpoint_root: str = "checkpoints", output_dir: str = "outputs", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.en_ckpt_base = os.path.join(checkpoint_root, "base_speakers", "EN")
        self.zh_ckpt_base = os.path.join(checkpoint_root, "base_speakers", "ZH")
        self.ckpt_converter = os.path.join(checkpoint_root, "converter")

        self.en_base_speaker_tts = None
        self.zh_base_speaker_tts = None
        self.tone_color_converter = None
        self.en_source_default_se = None
        self.en_source_style_se = None
        self.zh_source_se = None

        self.ready = False
        self.startup_message = "Service not initialized"

    def _required_files(self):
        return [
            os.path.join(self.en_ckpt_base, "config.json"),
            os.path.join(self.en_ckpt_base, "checkpoint.pth"),
            os.path.join(self.en_ckpt_base, "en_default_se.pth"),
            os.path.join(self.en_ckpt_base, "en_style_se.pth"),
            os.path.join(self.zh_ckpt_base, "config.json"),
            os.path.join(self.zh_ckpt_base, "checkpoint.pth"),
            os.path.join(self.zh_ckpt_base, "zh_default_se.pth"),
            os.path.join(self.ckpt_converter, "config.json"),
            os.path.join(self.ckpt_converter, "checkpoint.pth"),
        ]

    def bootstrap(self):
        missing = [pth for pth in self._required_files() if not os.path.exists(pth)]
        if missing:
            self.ready = False
            self.startup_message = "Missing required checkpoints:\n" + "\n".join(missing)
            return self.ready, self.startup_message

        try:
            self.en_base_speaker_tts = BaseSpeakerTTS(f"{self.en_ckpt_base}/config.json", device=self.device)
            self.en_base_speaker_tts.load_ckpt(f"{self.en_ckpt_base}/checkpoint.pth")
            self.zh_base_speaker_tts = BaseSpeakerTTS(f"{self.zh_ckpt_base}/config.json", device=self.device)
            self.zh_base_speaker_tts.load_ckpt(f"{self.zh_ckpt_base}/checkpoint.pth")
            self.tone_color_converter = ToneColorConverter(f"{self.ckpt_converter}/config.json", device=self.device)
            self.tone_color_converter.load_ckpt(f"{self.ckpt_converter}/checkpoint.pth")

            self.en_source_default_se = torch.load(f"{self.en_ckpt_base}/en_default_se.pth").to(self.device)
            self.en_source_style_se = torch.load(f"{self.en_ckpt_base}/en_style_se.pth").to(self.device)
            self.zh_source_se = torch.load(f"{self.zh_ckpt_base}/zh_default_se.pth").to(self.device)
        except Exception as exc:
            self.ready = False
            self.startup_message = f"Model bootstrap failed: {exc}"
            return self.ready, self.startup_message

        self.ready = True
        self.startup_message = f"OpenVoice model bootstrap successful on device={self.device}"
        return self.ready, self.startup_message

    def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        if not self.ready:
            return SynthesisResponse(False, f"[ERROR] Service is unavailable. {self.startup_message}")

        if request.agree is False:
            return SynthesisResponse(False, "[ERROR] Please accept the Terms & Condition!")

        language_predicted = langid.classify(request.prompt)[0].strip()
        if language_predicted not in SUPPORTED_LANGUAGES:
            return SynthesisResponse(
                False,
                f"[ERROR] The detected language {language_predicted} is not in supported languages: {SUPPORTED_LANGUAGES}",
            )

        if len(request.prompt) < 2:
            return SynthesisResponse(False, "[ERROR] Please give a longer prompt text")
        if len(request.prompt) > 200:
            return SynthesisResponse(
                False,
                "[ERROR] Text length limited to 200 characters for this demo.",
            )

        if language_predicted == "zh":
            if request.style != "default":
                return SynthesisResponse(False, "[ERROR] Chinese only supports 'default' style")
            tts_model = self.zh_base_speaker_tts
            source_se = self.zh_source_se
            language = "Chinese"
        else:
            if request.style not in ENGLISH_STYLES:
                return SynthesisResponse(False, f"[ERROR] Unsupported English style: {request.style}")
            tts_model = self.en_base_speaker_tts
            source_se = self.en_source_default_se if request.style == "default" else self.en_source_style_se
            language = "English"

        try:
            target_se, _ = se_extractor.get_se(
                request.audio_file_path,
                self.tone_color_converter,
                target_dir="processed",
                vad=True,
            )
        except Exception as exc:
            return SynthesisResponse(False, f"[ERROR] Get target tone color error: {exc}")

        src_path = f"{self.output_dir}/tmp.wav"
        tts_model.tts(request.prompt, src_path, speaker=request.style, language=language)

        save_path = f"{self.output_dir}/output.wav"
        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=save_path,
            message="@MyShell",
        )

        return SynthesisResponse(
            True,
            "Get response successfully",
            output_audio_path=save_path,
            reference_audio_path=request.audio_file_path,
        )
