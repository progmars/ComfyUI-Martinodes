import torch
import comfy.nested_tensor
from .latentops import unpack_samples
from .shared import CATEGORY


class LatentAVContrast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "av_latent": ("LATENT", ),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "interpolate": ("BOOLEAN", {"default": False}),
                "contrast_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "keep_norm": ( "BOOLEAN", { "default": True }),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("av_latent", )

    SEARCH_ALIASES = ["latent contrast", "adjust video latent contrast", "video latent contrast"]

    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Adjusts video latent contrast, optionally interpolating from contrast to contrast_end across time. Audio is unchanged."

    def run(self, av_latent, contrast, interpolate, contrast_end, keep_norm):
        samples = av_latent["samples"]

        tensors = unpack_samples(samples)

        orig = tensors[0].float()

        # Scale deviation from the per-channel mean to increase/decrease contrast.
        dims = tuple(d for d in range(2, orig.dim()))
        mean = orig.mean(dim=dims, keepdim=True)
        contrast_scale = contrast
        if interpolate:
            contrast_scale = torch.linspace(
                contrast,
                contrast_end,
                steps=orig.shape[2],
                device=orig.device,
                dtype=orig.dtype,
            ).view(1, 1, orig.shape[2], *([1] * (orig.dim() - 3)))
        video_tensor = mean + (orig - mean) * contrast_scale
        if keep_norm:
            # Preserve each token's own norm (channel dim)
            orig_norm = orig.norm(dim=1, keepdim=True)
            out_norm = video_tensor.norm(dim=1, keepdim=True)
            video_tensor = video_tensor * (orig_norm / out_norm.clamp_min(1e-6))

        tensors[0] = video_tensor.to(dtype=orig.dtype)

        out_av = av_latent.copy()
        if len(tensors) > 1:
            out_av["samples"] = comfy.nested_tensor.NestedTensor(tensors)
        else:
            out_av["samples"] = video_tensor

        return (out_av, )
