from .shared import CATEGORY, MINIMAX_H3_PARAMS as VIDEO_MODEL_PARAMS
from .latentops import unpack_samples


class LatentTokenExtractor:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", ),
                "token_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "frame_index", "frame_count")

    SEARCH_ALIASES = ["extract latent token", "pick latent frame", "latent token"]

    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Extracts a single temporal video latent token (1 or 4 frames for MiniMax H3) at the given token index. Also outputs the pixel frame index and frame count the token covers."

    def run(self, latent, token_index):
        if latent is None:
            return (latent, 0, 0)
        
        samples = latent.get("samples", None)
        if samples is None:
            raise ValueError("LatentTokenExtractor: latent does not contain 'samples'")

        tensors = unpack_samples(samples)
        if not tensors:
            raise ValueError("LatentTokenExtractor: No sample tensors found in latent")

        video_tensor = tensors[0]  # [B, C, T, H, W]
        total_t = video_tensor.shape[2]
        if token_index >= total_t:
            raise ValueError(f"LatentTokenExtractor: token_index {token_index} is out of range (total tokens: {total_t})")

        params = VIDEO_MODEL_PARAMS
        frame_multi = params["frame_multi"]
        frame_offset = params["frame_offset"]

        # token pattern per chunk: 1,4,4,4,4 frames
        chunk, pos = divmod(token_index, frame_offset)
        if pos == 0:
            frame_index = chunk * frame_multi
            frame_count = 1
        else:
            frame_index = chunk * frame_multi + 1 + (pos - 1) * 4
            frame_count = 4

        out = {"samples": video_tensor[:, :, token_index:token_index + 1].clone()}
        return (out, frame_index, frame_count)
