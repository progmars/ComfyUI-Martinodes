from .latentops import unpack_samples
from .shared import CATEGORY


class LatentAVSplitter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "av_latent": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("video_latent", "audio_latent")

    SEARCH_ALIASES = ["split latent", "split video audio latent", "separate av latent"]

    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Splits av_latent into separate video and audio latents."

    def run(self, av_latent):
        samples = av_latent["samples"]

        tensors = unpack_samples(samples)

        video_samples = tensors[0] if len(tensors) > 0 else None
        audio_samples = tensors[1] if len(tensors) > 1 else None

        video_latent = av_latent.copy()
        audio_latent = av_latent.copy()

        video_latent["samples"] = video_samples
        audio_latent["samples"] = audio_samples

        video_latent.pop("noise_mask", None)
        audio_latent.pop("noise_mask", None)

        return (video_latent, audio_latent)