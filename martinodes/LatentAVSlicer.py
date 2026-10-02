from .shared import CATEGORY, MINIMAX_H3_PARAMS
from .latentops import align_duration_to_tokens, align_offset_to_tokens
import torch
import comfy.nested_tensor


class LatentAVSlicer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "av_latent": ("LATENT", ),
                "start_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "end_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "end_offset_from": (["end", "start"], {"default": "start"}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("av_latent", )

    SEARCH_ALIASES = ["slice latent", "slice video audio latent", "trim latent", "cut latent", "latent slicer"]

    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Takes a slice of a video+audio latent using two time offsets aligned to the MiniMax H3 latent grid."

    def run(self, av_latent, start_offset_seconds=0.0, end_offset_seconds=0.0, end_offset_from="end", video_fps=24.0, **kwargs):
        # Support positional argument where video_fps was passed in 4th position
        if isinstance(end_offset_from, (int, float)):
            video_fps = float(end_offset_from)
            end_offset_from = kwargs.get("end_offset_from", "end")

        start_offset_seconds = kwargs.get("start_offset_seconds", kwargs.get("start_offset", start_offset_seconds))
        end_offset_seconds = kwargs.get("end_offset_seconds", kwargs.get("end_offset", end_offset_seconds))
        video_fps = kwargs.get("video_fps", video_fps)
        end_offset_from = kwargs.get("end_offset_from", end_offset_from)

        out_av = self.slice_av_latent(
            av_latent=av_latent,
            start_offset_seconds=start_offset_seconds,
            end_offset_seconds=end_offset_seconds,
            end_offset_from=end_offset_from,
            video_fps=video_fps,
            params=MINIMAX_H3_PARAMS
        )
        return (out_av, )


    def slice_av_latent(self, av_latent, start_offset_seconds=0.0, end_offset_seconds=0.0, end_offset_from="end", video_fps=24.0, params=MINIMAX_H3_PARAMS):
        if av_latent is None:
            return None

        samples = av_latent.get("samples", None)
        if samples is None:
            raise ValueError("LatentAVSlicer: av_latent does not contain 'samples'")
        
        is_nested = False
        if isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(samples, "is_nested", False):
            tensors = list(samples.unbind())
            is_nested = True
        elif isinstance(samples, (list, tuple)):
            tensors = list(samples)
        elif isinstance(samples, torch.Tensor):
            tensors = [samples]
        else:
            tensors = [samples]

        if not tensors:
            raise ValueError("LatentAVSlicer: No sample tensors found in av_latent")

        video_tensor = tensors[0]
        audio_tensor = tensors[1] if len(tensors) > 1 else None

        dims_v = len(video_tensor.shape)
        t_dim_v = 2 if dims_v >= 3 else 0
        total_v = video_tensor.shape[t_dim_v]

        dims_a = len(audio_tensor.shape)
        t_dim_a = -1
        total_a = audio_tensor.shape[t_dim_a]

        start_v, start_a = align_offset_to_tokens(start_offset_seconds, video_fps, params)
        start_v = min(start_v, total_v)
        if audio_tensor is not None:
            start_a = min(start_a, total_a)

        if end_offset_from == "start":
            end_v, end_a = align_duration_to_tokens(end_offset_seconds, video_fps, params)
            end_v = min(end_v, total_v)
            end_a = min(end_a, total_a)
        else:  # "end"
            end_v_trim, end_a_trim = align_offset_to_tokens(end_offset_seconds, video_fps, params)
            end_v = max(0, total_v - end_v_trim)
            end_a = max(0, total_a - end_a_trim)

        if start_v >= end_v:
            hint = " (if 'end_offset_seconds' was intended as an absolute timestamp from start, set end_offset_from='start')" if end_offset_from == "end" else ""
            raise ValueError(
                f"LatentAVSlicer: Calculated start video token ({start_v}) is >= end video token ({end_v}) "
                f"(total video tokens: {total_v}){hint}. Slicing would result in an empty latent."
            )

        if start_a >= end_a:
            hint = " (if 'end_offset_seconds' was intended as an absolute timestamp from start, set end_offset_from='start')" if end_offset_from == "end" else ""
            raise ValueError(
                f"LatentAVSlicer: Calculated start audio token ({start_a}) is >= end audio token ({end_a}) "
                f"(total audio tokens: {total_a}){hint}. Slicing would result in empty audio."
            )

        idx_v = [slice(None)] * dims_v
        idx_v[t_dim_v] = slice(start_v, end_v)
        sliced_video = video_tensor[tuple(idx_v)]

        idx_a = [slice(None)] * dims_a
        idx_a[t_dim_a] = slice(start_a, end_a)
        sliced_audio = audio_tensor[tuple(idx_a)]
        sliced_tensors = [sliced_video, sliced_audio]

        out_av = av_latent.copy()
        if is_nested:
            out_av["samples"] = comfy.nested_tensor.NestedTensor(sliced_tensors)
        elif isinstance(samples, (list, tuple)):
            out_av["samples"] = type(samples)(sliced_tensors)
        elif len(sliced_tensors) == 1:
            out_av["samples"] = sliced_tensors[0]
        else:
            out_av["samples"] = comfy.nested_tensor.NestedTensor(sliced_tensors)

        return out_av
        
