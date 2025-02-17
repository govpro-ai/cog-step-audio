from cog import BasePredictor, Input, Path
import torch
import torchaudio
from pget import pget_manifest, pget_url
from typing import Optional
from stepaudio import StepAudio
class Predictor(BasePredictor):

  def setup(self, weights: Optional[Path] = None):
    # pget_manifest('manifest.pget')
    if weights is not None:
        # Get the actual URL from the URLFile object
        weights_url = weights.url if hasattr(weights, 'url') else str(weights)
        # pget_url("https://replicate.delivery/" + weights_url, "weights.tar")
        assert False, "This model does not support training!"
    else:
        self.speaker_embedding = None

  def predict(self,
    text_prompt: str = Input(description="Text prompt (must provide either text or audio prompt)", default=None),
    audio_prompt: Path = Input(description="Audio prompt (must provide either text or audio prompt)", default=None),
    speech_mode: str = Input(description="Speech mode (currently supports Rap and Song Vocals)", default="conversation", choices=["conversation", "rap", "vocal"]),
    speed_ratio: float = Input(description="Speed ratio", default=1.0),
    volumn_ratio: float = Input(description="Volumn ratio (not a typo)", default=1.0),
  ) -> Path:
    assert ((text_prompt is not None) or (audio_prompt is not None)), "You must provide either a text or audio prompt!"

    model = StepAudio(
      tokenizer_path=f"weights/Step-Audio-Tokenizer",
      tts_path=f"weights/Step-Audio-TTS-3B",
      llm_path=f"weights/Step-Audio-Chat",
    )

    text = "Model did not provide a text output."
    if text_prompt is not None:
      print("Using text prompt.")
      text, audio, sr = model(
        [{"role": "user", "content": text_prompt}],
        "闫雨婷" if speech_mode == "conversation" else "闫雨婷RAP" if speech_mode == "rap" else "闫雨婷VOCAL",
        speed_ratio,
        volumn_ratio,
      )
      torchaudio.save("out.wav", audio, sr)
    else:
      print("Using audio prompt.")
      text, audio, sr = model(
        [{"role": "user", "content": {"type": "audio", "audio": str(audio_prompt)}}],
        "闫雨婷" if speech_mode == "conversation" else "闫雨婷RAP" if speech_mode == "rap" else "闫雨婷VOCAL",
        speed_ratio,
        volumn_ratio,
      )
      torchaudio.save("out.wav", audio, sr)

    print(text)
    return Path("out.wav")

