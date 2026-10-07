from .shared import CATEGORY, MINIMAX_H3_PARAMS
from .latentops import align_duration_to_tokens, unpack_samples, pack_samples
import torch
import torch.nn.functional as F


class LatentAVMaskedExtender:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_av_latent": ("LATENT", ),
                "mode": (["extend_tail", "prepend_head"], {"default": "extend_tail"}),
                "overlap_duration_seconds": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 60.0, "step": 0.1}),
                "video_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "video_mask_fade_seconds": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 60.0, "step": 0.1}),
                "audio_mask_fade_seconds": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 60.0, "step": 0.1}),
                "trim_freeze_tail": ("BOOLEAN", {"default": False}),
                "freeze_threshold": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 0.1, "step": 1e-5})
            },
            "optional": {
                "extendable_latent": ("LATENT", )
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("av_latent", )
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, target_av_latent, mode, overlap_duration_seconds, video_fps, video_mask_fade_seconds, 
            audio_mask_fade_seconds,
            trim_freeze_tail, 
            freeze_threshold,
            extendable_latent=None):

        if extendable_latent is None:
            return (target_av_latent, )
        
        params = MINIMAX_H3_PARAMS

        video_overlap_tokens, audio_overlap_tokens = align_duration_to_tokens(overlap_duration_seconds, video_fps, params)

        audio_token_rate = params["audio_token_rate"]
        frame_multi = params["frame_multi"]
        frame_offset = params["frame_offset"]

        video_fade_frames = video_mask_fade_seconds * video_fps
        video_fade_tokens = int(round(video_fade_frames * (frame_offset / frame_multi)))
        audio_fade_tokens = int(audio_mask_fade_seconds * audio_token_rate)

        return self.extend_masked(
            extendable_latent, target_av_latent, mode,
            video_overlap_tokens, audio_overlap_tokens, 
            video_fade_tokens, audio_fade_tokens,
            trim_freeze_tail, freeze_threshold
        )


    def extend_masked(self, extendable_latent, target_av_latent, mode, video_overlap_tokens, 
                      audio_overlap_tokens, video_fade_tokens, audio_fade_tokens,
                      trim_freeze_tail, freeze_threshold):
        
        out_av = target_av_latent.copy()
        
        old_any_samples = extendable_latent["samples"]
        new_v = target_av_latent["samples"]

        old_tensors = unpack_samples(old_any_samples)
        new_tensors = unpack_samples(new_v)

        if len(new_tensors) != 2:
            raise ValueError("LatentAVMaskedExtender: target latent must contain video and audio samples")

        new_video_tensor = new_tensors[0]
        new_audio_tensor = new_tensors[1]

        if len(old_tensors) == 2:
            old_video_tensor, old_audio_tensor = old_tensors
        elif len(old_tensors) == 1:
            source_tensor = old_tensors[0]
            source_modality = self.single_modality(source_tensor)
            video_time_dim = 2 if new_video_tensor.ndim >= 3 else 0

            if source_modality == "audio":
                old_video_tensor = new_video_tensor.narrow(video_time_dim, 0, 0)
                old_audio_tensor = source_tensor
                video_overlap_tokens = 0
            else:
                if source_tensor.shape[-2:] != new_video_tensor.shape[-2:]:
                    target_size = new_video_tensor.shape[-2:]
                    source_dtype = source_tensor.dtype
                    if source_tensor.ndim == 5:
                        batch_size, channels, frame_count, source_height, source_width = source_tensor.shape
                        source_frames = source_tensor.permute(0, 2, 1, 3, 4).reshape(
                            batch_size * frame_count, channels, source_height, source_width
                        )
                    else:
                        source_frames = source_tensor
                    resized_frames = F.interpolate(
                        source_frames.float(),
                        size=target_size,
                        mode="bilinear",
                        align_corners=False
                    ).to(dtype=source_dtype)
                    if source_tensor.ndim == 5:
                        source_tensor = resized_frames.reshape(
                            batch_size, frame_count, channels, *target_size
                        ).permute(0, 2, 1, 3, 4).contiguous()
                    else:
                        source_tensor = resized_frames
                old_video_tensor = source_tensor.unsqueeze(2) if source_tensor.ndim == 4 else source_tensor
                old_audio_tensor = new_audio_tensor.narrow(-1, 0, 0)
                audio_overlap_tokens = 0
        else:
            raise ValueError("LatentAVMaskedExtender: loaded latent must contain one modality or combined video and audio")

        dims_v = len(old_video_tensor.shape)
        t_dim_v = 2 if dims_v >= 3 else 0
        dims_a = len(old_audio_tensor.shape)
        t_dim_a = -1

        # ==========================================
        # 1. FREEZE FRAME DETECTION & REMOVAL
        # ==========================================
        if trim_freeze_tail and mode == "extend_tail":
            F_v_total = old_video_tensor.shape[t_dim_v]
            freeze_count = 0
            
            # Step backward through the latent temporal dimension
            for i in range(F_v_total - 1, 0, -1):
                idx_curr = [slice(None)] * dims_v
                idx_prev = [slice(None)] * dims_v
                idx_curr[t_dim_v] = i
                idx_prev[t_dim_v] = i - 1
                
                curr_frame = old_video_tensor[tuple(idx_curr)]
                prev_frame = old_video_tensor[tuple(idx_prev)]
                
                # Mean Absolute Error (MAE) quantifies token-to-token variance.
                diff = torch.mean(torch.abs(curr_frame - prev_frame)).item()
                if diff <= freeze_threshold:
                    freeze_count += 1
                else:
                    break
            
            if freeze_count > 0:
                # Trim frozen video tokens
                idx_trim_v = [slice(None)] * dims_v
                idx_trim_v[t_dim_v] = slice(None, -freeze_count)
                old_video_tensor = old_video_tensor[tuple(idx_trim_v)]
                
                # Trim proportional audio tokens
                F_a_total = old_audio_tensor.shape[t_dim_a]
                
                # Assuming uniform temporal compression, trimming audio tokens proportionally 
                # to the discarded video tokens is required to maintain AV synchronization.
                audio_trim_count = int(round(freeze_count * (F_a_total / F_v_total)))
                
                if audio_trim_count > 0:
                    idx_trim_a = [slice(None)] * dims_a
                    idx_trim_a[t_dim_a] = slice(None, -audio_trim_count)
                    old_audio_tensor = old_audio_tensor[tuple(idx_trim_a)]

                print(f"[LatentAVMaskedExtender] Freeze detected. Trimmed {freeze_count} video tokens and {audio_trim_count} audio tokens. Threshold: {freeze_threshold}.")
            else:
                print(f"[LatentAVMaskedExtender] Freezes not detected.")
                
        # Ensure we do not request more overlap than the newly trimmed tensor contains
        video_overlap_tokens = min(video_overlap_tokens, old_video_tensor.shape[t_dim_v])
        audio_overlap_tokens = min(audio_overlap_tokens, old_audio_tensor.shape[t_dim_a])

        # ==========================================
        # 2. VIDEO PROCESSING
        # ==========================================
        idx_overlap_old_v = [slice(None)] * dims_v
        idx_rem_new_v = [slice(None)] * dims_v

        if mode == "extend_tail":
            idx_overlap_old_v[t_dim_v] = slice(-video_overlap_tokens, None) if video_overlap_tokens > 0 else slice(0, 0)
            idx_rem_new_v[t_dim_v] = slice(video_overlap_tokens, None)
            overlap_v = old_video_tensor[tuple(idx_overlap_old_v)]
            rem_v = new_video_tensor[tuple(idx_rem_new_v)]
            combined_v = torch.cat((overlap_v, rem_v), dim=t_dim_v)
        else: # prepend_head
            idx_overlap_old_v[t_dim_v] = slice(None, video_overlap_tokens) if video_overlap_tokens > 0 else slice(0, 0)
            idx_rem_new_v[t_dim_v] = slice(None, -video_overlap_tokens) if video_overlap_tokens > 0 else slice(None)
            overlap_v = old_video_tensor[tuple(idx_overlap_old_v)]
            rem_v = new_video_tensor[tuple(idx_rem_new_v)]
            combined_v = torch.cat((rem_v, overlap_v), dim=t_dim_v)
        
        F_v = combined_v.shape[t_dim_v]
        H_v = combined_v.shape[-2]
        W_v = combined_v.shape[-1]
        
        mask_v = torch.ones((F_v, H_v, W_v), dtype=torch.float32, device=combined_v.device)
        video_fade_tokens = min(video_fade_tokens, video_overlap_tokens)

        # for experiments
        mask_min = 0.0
        mask_max = 1.0

        # smoothly interpolate to avoid sharp transitions (which sometimes cause scene cuts)
        # between
        # 0 - keep the original latent
        # 1 - regenerate with new data

        if mode == "extend_tail":
            if video_overlap_tokens > video_fade_tokens:
                # keep all by default
                mask_v[:(video_overlap_tokens - video_fade_tokens), :, :] = mask_min
            if video_fade_tokens > 0:
                fade_gradient_v = torch.linspace(mask_min, mask_max, video_fade_tokens, dtype=torch.float32, device=combined_v.device)
                mask_v[(video_overlap_tokens - video_fade_tokens):video_overlap_tokens, :, :] = fade_gradient_v.view(-1, 1, 1)
        else:
            r_len = F_v - video_overlap_tokens
            if video_overlap_tokens > video_fade_tokens:
                mask_v[(r_len + video_fade_tokens):, :, :] = mask_min
            if video_fade_tokens > 0:
                if video_fade_tokens == 1:
                    fade_gradient_v = torch.full((1,), mask_min, dtype=torch.float32, device=combined_v.device)
                else:
                    fade_gradient_v = torch.linspace(mask_max, mask_min, video_fade_tokens, dtype=torch.float32, device=combined_v.device)
                mask_v[r_len:(r_len + video_fade_tokens), :, :] = fade_gradient_v.view(-1, 1, 1)
        
        mask_v_reshaped = mask_v.reshape((combined_v.shape[0] * F_v, 1, H_v, W_v))

        # ==========================================
        # 3. AUDIO PROCESSING
        # ==========================================
        idx_overlap_old_a = [slice(None)] * dims_a
        idx_rem_new_a = [slice(None)] * dims_a

        if mode == "extend_tail":
            idx_overlap_old_a[t_dim_a] = slice(-audio_overlap_tokens, None) if audio_overlap_tokens > 0 else slice(0, 0)
            idx_rem_new_a[t_dim_a] = slice(audio_overlap_tokens, None)
            overlap_a = old_audio_tensor[tuple(idx_overlap_old_a)]
            rem_a = new_audio_tensor[tuple(idx_rem_new_a)]
            combined_a = torch.cat((overlap_a, rem_a), dim=t_dim_a)
        else: # prepend_head
            idx_overlap_old_a[t_dim_a] = slice(None, audio_overlap_tokens) if audio_overlap_tokens > 0 else slice(0, 0)
            idx_rem_new_a[t_dim_a] = slice(None, -audio_overlap_tokens) if audio_overlap_tokens > 0 else slice(None)
            overlap_a = old_audio_tensor[tuple(idx_overlap_old_a)]
            rem_a = new_audio_tensor[tuple(idx_rem_new_a)]
            combined_a = torch.cat((rem_a, overlap_a), dim=t_dim_a)
        
        B_a = combined_a.shape[0]
        Freq_a = combined_a.shape[-2]
        Time_a = combined_a.shape[-1]
        
        mask_a = torch.ones((B_a, Freq_a, Time_a), dtype=torch.float32, device=combined_a.device)
        audio_fade_tokens = min(audio_fade_tokens, audio_overlap_tokens)
        
        if mode == "extend_tail":
            if audio_overlap_tokens > audio_fade_tokens:
                # keep all by default
                mask_a[:, :, :(audio_overlap_tokens - audio_fade_tokens)] = mask_min
            if audio_fade_tokens > 0:
                fade_gradient_a = torch.linspace(mask_min, mask_max, audio_fade_tokens, dtype=torch.float32, device=combined_a.device)
                mask_a[:, :, (audio_overlap_tokens - audio_fade_tokens):audio_overlap_tokens] = fade_gradient_a.view(1, 1, -1)
        else:
            r_len_a = Time_a - audio_overlap_tokens
            if audio_overlap_tokens > audio_fade_tokens:
                mask_a[:, :, (r_len_a + audio_fade_tokens):] = mask_min
            if audio_fade_tokens > 0:
                fade_gradient_a = torch.linspace(mask_max, mask_min, audio_fade_tokens, dtype=torch.float32, device=combined_a.device)
                mask_a[:, :, r_len_a:(r_len_a + audio_fade_tokens)] = fade_gradient_a.view(1, 1, -1)
        
        mask_a_reshaped = mask_a.reshape((B_a, 1, Freq_a, Time_a))

        # ==========================================
        # 4. RECONSTRUCT OUTPUTS
        # ==========================================

        tensors_out = [combined_v, combined_a]
        masks_out = [mask_v_reshaped, mask_a_reshaped]

        # ==========================================
        # 5. MERGE MASKS
        # ==========================================

        existing_masks = target_av_latent.get("noise_mask")
        if existing_masks is not None:
            existing_masks = unpack_samples(existing_masks)

            if existing_masks:
                batch_size = new_video_tensor.shape[0] if new_video_tensor.ndim > 1 else 1
                old_video_mask = existing_masks[0]
                if isinstance(old_video_mask, torch.Tensor):
                    if old_video_mask.ndim == 4 and old_video_mask.shape[1] == 1:
                        old_video_mask = old_video_mask.reshape(
                            batch_size, -1, 1, old_video_mask.shape[-2], old_video_mask.shape[-1]
                        )
                    elif old_video_mask.ndim == 4:
                        old_video_mask = old_video_mask.unsqueeze(2)
                    elif old_video_mask.ndim == 5 and old_video_mask.shape[1] == 1:
                        old_video_mask = old_video_mask.permute(0, 2, 1, 3, 4)

                    new_mask_v = mask_v.reshape(batch_size, F_v, 1, H_v, W_v)
                    if old_video_mask.shape == new_mask_v.shape:
                        old_video_mask = old_video_mask.to(device=mask_v.device, dtype=mask_v.dtype)
                        new_mask_v = torch.where(new_mask_v == mask_max, old_video_mask, new_mask_v)
                        mask_v_reshaped = new_mask_v.reshape_as(mask_v_reshaped)

                if len(existing_masks) > 1:
                    old_audio_mask = existing_masks[1]
                    if isinstance(old_audio_mask, torch.Tensor):
                        if old_audio_mask.ndim == 4 and old_audio_mask.shape[1] == 1:
                            old_audio_mask = old_audio_mask.squeeze(1)
                        if old_audio_mask.ndim == 3:
                            old_audio_mask = old_audio_mask.to(device=mask_a.device, dtype=mask_a.dtype)
                            if old_audio_mask.shape == mask_a.shape:
                                mask_a = torch.where(mask_a == mask_max, old_audio_mask, mask_a)
                                mask_a_reshaped = mask_a.reshape_as(mask_a_reshaped)

                masks_out = [mask_v_reshaped, mask_a_reshaped]
        
        out_av["samples"] = pack_samples(new_v, tensors_out)
        out_av["noise_mask"] = pack_samples(new_v, masks_out)
                
        return (out_av, )

    @staticmethod
    def single_modality(tensor):
        if tensor.ndim == 5:
            return "video"
        if tensor.ndim == 4:
            return "audio" if tensor.shape[1] == 8 else "image"
        raise ValueError("LatentAVMaskedExtender: single-modality samples must be 4D or 5D tensors")

