def align_time_to_tokens(seconds, video_fps, params):
    if seconds <= 0:
        return 0, 0
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

