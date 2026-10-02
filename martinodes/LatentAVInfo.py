import torch
import comfy.nested_tensor
from .shared import CATEGORY, MINIMAX_H3_PARAMS


class LatentAVInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "av_latent": ("LATENT", {}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("av_latent", "frames", "video_resolution", "video_tokens", "audio_tokens", "mask_info")

    SEARCH_ALIASES = ["av latent info", "latent info", "video audio latent info"]

    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Displays basic info for a combined AV latent: frame count, video resolution, token counts, and mask information."

    def run(self, **kwargs):
        av_latent = kwargs.get("av_latent", None)
        if av_latent is None:
            raise Exception("No av_latent provided.")

        samples = av_latent.get("samples", None)
        if samples is None:
            raise Exception("av_latent is missing samples.")

        if isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(samples, "is_nested", False):
            tensors = list(samples.unbind())
        elif isinstance(samples, (list, tuple)):
            tensors = list(samples)
        elif isinstance(samples, torch.Tensor):
            tensors = [samples]
        else:
            tensors = [samples]

        if len(tensors) < 2:
            raise Exception("av_latent does not contain both video and audio samples.")

        video_tensor = tensors[0]
        audio_tensor = tensors[1]

        dims_v = len(video_tensor.shape)
        t_dim_v = 2 if dims_v >= 3 else 0
        video_tokens = int(video_tensor.shape[t_dim_v])
        audio_tokens = int(audio_tensor.shape[-1])

        latent_h = int(video_tensor.shape[-2])
        latent_w = int(video_tensor.shape[-1])
        video_resolution = f"{latent_w}x{latent_h}"

        # Inverse of the token snapping rule used by MiniMax H3 latent nodes.
        frame_multi = float(MINIMAX_H3_PARAMS["frame_multi"])
        frame_offset = float(MINIMAX_H3_PARAMS["frame_offset"])
        min_latent_temporal_tokens = float(MINIMAX_H3_PARAMS["min_latent_temporal_tokens"])

        if video_tokens <= min_latent_temporal_tokens:
            frames = int(round(frame_offset))
        else:
            k = (float(video_tokens) - min_latent_temporal_tokens) / frame_offset
            frames = int(round(k * frame_multi + frame_offset))

        # Latent mask information
        noise_mask = av_latent.get("noise_mask", None)
        if noise_mask is None:
            mask_info = "None"
        else:
            if isinstance(noise_mask, comfy.nested_tensor.NestedTensor) or getattr(noise_mask, "is_nested", False):
                mask_tensors = list(noise_mask.unbind())
            elif isinstance(noise_mask, (list, tuple)):
                mask_tensors = list(noise_mask)
            elif isinstance(noise_mask, torch.Tensor):
                mask_tensors = [noise_mask]
            else:
                mask_tensors = [noise_mask]

            if len(mask_tensors) == 0:
                mask_info = "None"
            else:
                parts = []
                for idx, m in enumerate(mask_tensors):
                    if not isinstance(m, torch.Tensor):
                        parts.append(f"item_{idx}: {type(m).__name__}")
                        continue

                    if m.numel() == 0:
                        parts.append(f"item_{idx}: empty")
                        continue

                    if len(mask_tensors) == 2:
                        label = "video" if idx == 0 else "audio"
                    elif len(mask_tensors) == 1:
                        label = "video"
                    else:
                        label = f"mask_{idx}"

                    min_val = round(float(m.min().item()), 3)
                    max_val = round(float(m.max().item()), 3)
                    val_str = f"all {min_val}" if min_val == max_val else f"range: {min_val}..{max_val}"
                    parts.append(f"{label}: {list(m.shape)} ({val_str})")

                mask_info = ", ".join(parts) if parts else "None"

        return (av_latent, frames, video_resolution, video_tokens, audio_tokens, mask_info)