from .shared import MINIMAX_H3_PARAMS
from .latentops import unpack_samples

class MinimaxH3KeyframeConditioningInjector:

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

        params = MINIMAX_H3_PARAMS # TODO: could support other models in the future as well
        frame_multi = int(params.get("frame_multi"))
        frame_offset = int(params.get("frame_offset"))
        min_latent_tokens = int(params.get("min_latent_temporal_tokens")) 
        keyframe_carrier_key = params.get("keyframe_carrier_key")

        # assume the latent has already been prepared in correct rounded length
        if latent_t <= min_latent_tokens:
            frame_count = frame_offset
        else:
            k = (latent_t - min_latent_tokens) // frame_offset
            frame_count = k * frame_multi + frame_offset
            
        last_frame_index = max(0, frame_count - 1)

        out_conditioning = inject_keyframe(conditioning, first_video, keyframe_carrier_key, 0)
        out_conditioning = inject_keyframe(out_conditioning, last_video, keyframe_carrier_key, last_frame_index)

        return (out_conditioning,)

    
class MinimaxH3AnyFrameConditioningInjector:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "frame_latent": ("LATENT", ),                
                "frame_index": ("INT", ),
             }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = "martinodes"


    def run(self, conditioning, frame_latent, frame_index):
        if conditioning is None or frame_latent is None:
            return (conditioning,)

        # no resize - assume the user has preprocessed the input frames
        # to required resolution
        injectable = None
        if frame_latent is not None and "samples" in frame_latent:
            first_list = unpack_samples(frame_latent["samples"])
            if first_list and len(first_list) > 0:
                injectable = first_list[0].float()

        if injectable is None:
            return (conditioning,)

        params = MINIMAX_H3_PARAMS # TODO: could support other models in the future as well
        keyframe_carrier_key = params.get("keyframe_carrier_key")

        out_conditioning = inject_keyframe(conditioning, injectable, keyframe_carrier_key, frame_index)

        return (out_conditioning,)


    
def inject_keyframe(conditioning, latent, frame_key_name, frame_index):
    
    out_conditioning = []
        
    for item in conditioning:
        cond_tensor, extra_dict = item
        new_extra_dict = extra_dict.copy()
        
        if frame_key_name not in new_extra_dict:
            new_extra_dict[frame_key_name] = []
        else:
            new_extra_dict[frame_key_name] = list(new_extra_dict[frame_key_name])

        if latent is not None:
            payload = {
                "resolved_frame_index": frame_index,
                "latent": latent.to(device=cond_tensor.device, dtype=cond_tensor.dtype)
            }
            new_extra_dict[frame_key_name].append(payload)

        out_conditioning.append([cond_tensor, new_extra_dict])

    return out_conditioning