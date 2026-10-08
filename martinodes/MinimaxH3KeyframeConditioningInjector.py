import torch.nn.functional as F

from .shared import MINIMAX_H3_PARAMS
from .latentops import unpack_samples

class MinimaxH3KeyframeConditioningInjector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "reference_av_latent": ("LATENT", ),
             },
            "optional": {
                "first_frame_latent": ("LATENT", ),
                "last_frame_latent": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "martinodes"


    def run(self, conditioning, reference_av_latent, first_frame_latent=None, last_frame_latent=None):
        if conditioning is None or reference_av_latent is None:
            return (conditioning,)

        target_list = unpack_samples(reference_av_latent["samples"])
        if not target_list or len(target_list) == 0:
            return (conditioning,)
            
        target_video_tensor = target_list[0]
        target_shape = target_video_tensor.shape # [B, C, latent_t, H, W]
        latent_t = target_shape[2]

        # no resize - assume the user has preprocessed the input frames
        # to required resolution
        first_video = None
        if first_frame_latent is not None and "samples" in first_frame_latent:
            first_list = unpack_samples(first_frame_latent["samples"])
            if first_list and len(first_list) > 0:
                first_video = first_list[0].float()

        last_video = None
        if last_frame_latent is not None and "samples" in last_frame_latent:
            last_list = unpack_samples(last_frame_latent["samples"])
            if last_list and len(last_list) > 0:
                last_video = last_list[0].float()

        if first_video is None and last_video is None:
            return (conditioning,)

        params = MINIMAX_H3_PARAMS
        frame_multi = int(params.get("frame_multi"))
        frame_offset = int(params.get("frame_offset"))
        min_latent_tokens = int(params.get("min_latent_temporal_tokens")) 

        # assume the latent has already been prepared in correct rounded length
        if latent_t <= min_latent_tokens:
            frame_count = frame_offset
        else:
            k = (latent_t - min_latent_tokens) // frame_offset
            frame_count = k * frame_multi + frame_offset
            
        last_frame_index = max(0, frame_count - 1)

        out_conditioning = []
        
        for item in conditioning:
            cond_tensor, extra_dict = item
            new_extra_dict = extra_dict.copy()
            
            if "minimax_keyframes" not in new_extra_dict:
                new_extra_dict["minimax_keyframes"] = []
            else:
                new_extra_dict["minimax_keyframes"] = list(new_extra_dict["minimax_keyframes"])

            if first_video is not None:
                first_payload = {
                    "resolved_frame_index": 0,
                    "latent": first_video.to(device=cond_tensor.device, dtype=cond_tensor.dtype)
                }
                new_extra_dict["minimax_keyframes"].append(first_payload)

            if last_video is not None:
                last_payload = {
                    "resolved_frame_index": last_frame_index,
                    "latent": last_video.to(device=cond_tensor.device, dtype=cond_tensor.dtype)
                }
                new_extra_dict["minimax_keyframes"].append(last_payload)
            
            out_conditioning.append([cond_tensor, new_extra_dict])

        return (out_conditioning,)