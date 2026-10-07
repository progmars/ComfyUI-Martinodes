import torch
from .latentops import unpack_samples
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

    @staticmethod
    def describe_mask_shape(mask, label, batch_size, video_tokens):
        shape = mask.shape
        if label == "video":
            if mask.ndim == 4 and shape[0] == batch_size * video_tokens and shape[1] == 1:
                return (
                    f"batch={batch_size}, video_tokens={video_tokens} (flattened with batch), "
                    f"channels={shape[1]}, latent_resolution={shape[3]}x{shape[2]}"
                )
            if mask.ndim == 4 and shape[0] == batch_size and shape[1] == video_tokens:
                return (
                    f"batch={shape[0]}, video_tokens={shape[1]}, "
                    f"latent_resolution={shape[3]}x{shape[2]} (no channel axis)"
                )
            if mask.ndim == 5 and shape[0] == batch_size and shape[1] == video_tokens:
                return (
                    f"batch={shape[0]}, video_tokens={shape[1]}, channels={shape[2]}, "
                    f"latent_resolution={shape[4]}x{shape[3]}"
                )
            if mask.ndim == 5 and shape[0] == batch_size and shape[2] == video_tokens:
                return (
                    f"batch={shape[0]}, channels={shape[1]}, video_tokens={shape[2]}, "
                    f"latent_resolution={shape[4]}x{shape[3]}"
                )
        elif label == "audio":
            if mask.ndim == 4:
                return (
                    f"batch={shape[0]}, channels={shape[1]}, frequency_bins={shape[2]}, "
                    f"audio_tokens={shape[3]}"
                )
            if mask.ndim == 3:
                return f"batch={shape[0]}, frequency_bins={shape[1]}, audio_tokens={shape[2]}"
            if mask.ndim == 2:
                return f"frequency_bins={shape[0]}, audio_tokens={shape[1]}"

        return ", ".join(f"axis{axis}={size}" for axis, size in enumerate(shape))

    @staticmethod
    def mask_temporal_values(mask, label, batch_size, video_tokens, audio_tokens):
        if label == "video":
            if mask.ndim == 4:
                if mask.shape[1] == 1 and mask.shape[0] == batch_size * video_tokens:
                    mask = mask.reshape(batch_size, video_tokens, 1, *mask.shape[-2:])
                elif mask.shape[0] == batch_size and mask.shape[1] == video_tokens:
                    mask = mask.unsqueeze(2)
                else:
                    return None
            elif mask.ndim == 5:
                if mask.shape[0] != batch_size:
                    return None
                if mask.shape[1] == 1 and mask.shape[2] == video_tokens:
                    mask = mask.permute(0, 2, 1, 3, 4)
                elif mask.shape[1] != video_tokens:
                    return None
            else:
                return None
            time_axis = 1
            time_tokens = video_tokens
        elif label == "audio" and mask.ndim >= 2:
            time_axis = mask.ndim - 1
            time_tokens = audio_tokens
        else:
            return None

        if mask.shape[time_axis] != time_tokens:
            return None

        per_time = mask.movedim(time_axis, -1).reshape(-1, time_tokens)
        return per_time.amin(dim=0).tolist(), per_time.amax(dim=0).tolist()

    @staticmethod
    def describe_mask_regions(mask, label, batch_size, video_tokens, audio_tokens):
        temporal_values = LatentAVInfo.mask_temporal_values(
            mask, label, batch_size, video_tokens, audio_tokens
        )
        if temporal_values is None:
            return None

        minima, maxima = temporal_values
        regions = []
        start = 0

        def category(min_value, max_value):
            if max_value <= 1e-6:
                return "keep"
            if min_value >= 1.0 - 1e-6:
                return "regenerate"
            return "blend"

        while start < len(minima):
            region_category = category(minima[start], maxima[start])
            end = start + 1
            while end < len(minima) and category(minima[end], maxima[end]) == region_category:
                end += 1

            region_min = min(minima[start:end])
            region_max = max(maxima[start:end])
            unit = "frames" if label == "video" else "tokens"
            value_info = (
                f"mask {region_min:.3g}"
                if region_min == region_max
                else f"mask {region_min:.3g}..{region_max:.3g}"
            )
            regions.append(
                f"{region_category} {unit} {start}-{end - 1} ({value_info})"
            )
            start = end

        return "; ".join(regions) if regions else "empty"


    def run(self, **kwargs):
        av_latent = kwargs.get("av_latent", None)
        if av_latent is None:
            raise Exception("No av_latent provided.")

        samples = av_latent.get("samples", None)
        if samples is None:
            raise Exception("av_latent is missing samples.")
        
        tensors = unpack_samples(samples)

        if len(tensors) < 2:
            raise Exception("av_latent does not contain both video and audio samples.")

        video_tensor = tensors[0]
        audio_tensor = tensors[1]

        dims_v = len(video_tensor.shape)
        t_dim_v = 2 if dims_v >= 3 else 0
        video_tokens = int(video_tensor.shape[t_dim_v])
        audio_tokens = int(audio_tensor.shape[-1])
        batch_size = video_tensor.shape[0] if video_tensor.ndim > 1 else 1

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
            mask_tensors = unpack_samples(noise_mask)
            
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
                    region_info = self.describe_mask_regions(
                        m, label, batch_size, video_tokens, audio_tokens
                    )
                    shape_info = self.describe_mask_shape(m, label, batch_size, video_tokens)
                    region_str = f", regions: {region_info}" if region_info is not None else ""
                    parts.append(f"{label}: {shape_info} ({val_str}{region_str})")

                mask_info = ", ".join(parts) if parts else "None"

        return (av_latent, frames, video_resolution, video_tokens, audio_tokens, mask_info)