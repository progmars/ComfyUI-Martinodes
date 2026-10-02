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
    Aligns time duration to video and audio token count (e.g. 5k - 3 for k >= 1).
    Used for standalone clips or sequence lengths (overlap etc).
    """
    audio_token_rate = params["audio_token_rate"]
    frame_multi = params["frame_multi"]
    frame_offset = params["frame_offset"]
    min_latent_temporal_tokens = params["min_latent_temporal_tokens"]

    target_frames = seconds * video_fps
    k = max(0, int(round((target_frames - frame_offset) / frame_multi)))
    snapped_video_frames = int(k * frame_multi + frame_offset)

    video_tokens = int(k * frame_offset + min_latent_temporal_tokens)
    audio_tokens = int(round(snapped_video_frames * (audio_token_rate / video_fps)))
    return video_tokens, audio_tokens


