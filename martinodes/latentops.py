import torch
import comfy.nested_tensor

def align_offset_to_tokens(seconds, video_fps, params):
    """
    Calculates the number of video and audio tokens to advance or trim by whole VAE chunks.
    Video VAE encodes in frame_multi-frame clips, each advancing the latent by frame_offset tokens.
    Slicing at head/tail must advance by multiples of frame_offset tokens to keep VAE temporal decoding in-phase.
    """
    audio_token_rate = params["audio_token_rate"]
    frame_multi = params["frame_multi"]
    frame_offset = params["frame_offset"]

    target_frames = seconds * video_fps
    k = max(0, int(round(target_frames / frame_multi)))
    video_tokens = int(k * frame_offset)
    audio_tokens = int(round((k * frame_multi) * (audio_token_rate / video_fps)))
    return video_tokens, audio_tokens


def align_duration_to_tokens(seconds, video_fps, params):
    """
    Aligns time duration to video and audio token count.
    It includes the frame offset to support standalone clips
    and snaps down (e.g. 1 second for 17k + 5 gives 22 frames),
    unlike the latent allocation in usual Comfy workflows
    that use formula max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17
    thus snapping up (1 second to 39 frames = 1.625 sec).

    For MiniMax H3 latent, the frame pattern is
    1,4,4,4,4,1,4,4,4,4...1,4,4,4,4,1,4
    So, it always ends with the 5 frame offset of 1,4,
    which is actually not offset from the latent start but rather suffix at the end.

    This function takes care to extract correctly aligned piece
    of at least one 1,4 segment (minimum tokens)
    or aligned 1,4,4,4,4,(repeat n times),1,4 segment.
    """
    audio_token_rate = params["audio_token_rate"] # 40
    frame_multi = params["frame_multi"] # 17
    frame_offset = params["frame_offset"] # 5
    min_latent_temporal_tokens = params["min_latent_temporal_tokens"] # 2

    target_frames = seconds * video_fps
    k = max(0, int(round((target_frames - frame_offset) / frame_multi)))
    snapped_video_frames = int(k * frame_multi + frame_offset)

    # print(f"snapped_video_frames: {snapped_video_frames}")

    video_tokens = int(k * frame_offset + min_latent_temporal_tokens)
    audio_tokens = int(round(snapped_video_frames * (audio_token_rate / video_fps)))
    return video_tokens, audio_tokens


def unpack_samples(samples):
    if isinstance(samples, comfy.nested_tensor.NestedTensor) or getattr(samples, "is_nested", False):
        return list(samples.unbind())
    if isinstance(samples, (list, tuple)):
        return list(samples)
    if isinstance(samples, torch.Tensor):
        return [samples]
    raise TypeError(f"LatentAVMaskedExtender: unsupported samples container {type(samples).__name__}")


def pack_samples(template, tensors):
    if isinstance(template, comfy.nested_tensor.NestedTensor) or getattr(template, "is_nested", False):
        return comfy.nested_tensor.NestedTensor(tensors)
    if isinstance(template, tuple):
        return tuple(tensors)
    if isinstance(template, list):
        return list(tensors)
    if len(tensors) == 1:
        return tensors[0]
    return comfy.nested_tensor.NestedTensor(tensors)

