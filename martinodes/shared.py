import comfy.audio
import torch

CATEGORY = "martinodes"

MINIMAX_H3_PARAMS = {
    "audio_token_rate": 40,
    "frame_multi": 17.0,
    "frame_offset": 5.0,
    "min_latent_temporal_tokens": 2.0,
    "keyframe_carrier_key": "minimax_keyframes"
}


def list_head_tail(list, take_count, start_end):

    total = list.shape[0]
    take_count = min(take_count, total)

    if start_end == "start":
        new_list = list[:take_count]
    else:
        new_list = list[-take_count:]

    return new_list


def get_batch_size(batch):
    if isinstance(batch, (list, tuple)):
        return len(batch)

    if isinstance(batch, dict):
        for key in ("samples", "waveform", "images"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
                return value.shape[0]
        raise TypeError("Expected a batch dictionary with samples, waveform, or images.")

    if hasattr(batch, "shape") and len(batch.shape) > 0:
        return batch.shape[0]

    x = type(batch)
    raise TypeError(f"Expected a list, tuple, tensor, or ComfyUI batch dictionary, got {x}")


def unwrap_singleton_batch_input(batch):
    if isinstance(batch, list) and len(batch) == 1 and not isinstance(batch[0], str):
        return batch[0]
    return batch


def select_batch_item(batch, index, batch_size):
    if isinstance(batch, list):
        return batch[index]

    if isinstance(batch, tuple):
        return batch[index]

    if isinstance(batch, dict):
        selected = batch.copy()
        for key in ("samples", "waveform", "images", "noise_mask"):
            value = batch.get(key)
            if value is not None and hasattr(value, "shape") and len(value.shape) > 0:
                if value.shape[0] == batch_size:
                    selected[key] = value[index:index + 1]

        batch_index = batch.get("batch_index")
        if isinstance(batch_index, (list, tuple)) and len(batch_index) == batch_size:
            selected["batch_index"] = batch_index[index:index + 1]
        return selected

    return batch[index:index + 1]


def resample_audio(audio, samplerate, channels):
    waveform = audio["waveform"]
    orig_sr = audio["sample_rate"]

    if orig_sr != samplerate:
        waveform = comfy.audio.resample(waveform, orig_sr, samplerate)

    current_channels = waveform.shape[1]
    if current_channels > channels:
        waveform = waveform[:, :channels, :]
    elif current_channels < channels:
        last_channel = waveform[:, -1:, :].expand(-1, channels - current_channels, -1)
        waveform = torch.cat((waveform, last_channel), dim=1)

    out_audio = {
        "waveform": waveform,
        "sample_rate": samplerate
    }

    return out_audio
