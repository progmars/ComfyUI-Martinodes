from .martinodes.MediaSlicer import MediaSlicer
from .martinodes.AudioResampler import AudioResampler
from .martinodes.AudioTrimExtender import AudioTrimExtender
from .martinodes.AudioInfo import AudioInfo
from .martinodes.LatentAVInfo import LatentAVInfo
from .martinodes.MediaTrimmer import MediaTrimmer
from .martinodes.MediaOverlappingConcatenator import MediaOverlappingConcatenator
from .martinodes.LatentAVMaskedExtender import LatentAVMaskedExtender
from .martinodes.LatentAVLoadSave import NoneLatent, SaveAVLatent, LoadAVLatent
from .martinodes.LatentOverlappingConcatenator import LatentOverlappingConcatenator, LatentFolderOverlappingConcatenator
from .martinodes.LatentAVCombiner import LatentAVCombiner
from .martinodes.LatentAVSplitter import LatentAVSplitter
from .martinodes.LatentAVContrast import LatentAVContrast
from .martinodes.VideoLatentPicker import VideoLatentPicker

NODE_CLASS_MAPPINGS = {
    "MARMediaSlicer": MediaSlicer,
    "MARAudioResampler": AudioResampler,
    "MARAudioTrimExtender": AudioTrimExtender,
    "MARMediaTrimmer": MediaTrimmer,
    "MARAudioInfo": AudioInfo,
    "LatentAVInfo": LatentAVInfo,
    "MARMediaOverlappingConcatenator": MediaOverlappingConcatenator,
    "LatentAVMaskedExtender": LatentAVMaskedExtender,
    "NoneLatent": NoneLatent,
    "SaveAVLatent": SaveAVLatent,
    "LoadAVLatent": LoadAVLatent,
    "LatentOverlappingConcatenator": LatentOverlappingConcatenator,
    "LatentFolderOverlappingConcatenator": LatentFolderOverlappingConcatenator,
    "LatentAVCombiner": LatentAVCombiner,
    "LatentAVSplitter": LatentAVSplitter,
    "LatentAVContrast": LatentAVContrast,
    "VideoLatentPicker": VideoLatentPicker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MARMediaSlicer": "Media head or tail",
    "MARAudioInfo": "Audio info",
    "MARAudioResampler": "Resample audio",
    "MARMediaTrimmer": "Trim media",
    "MARAudioTrimExtender": "Ensure audio duration",
    "LatentAVInfo": "Video+audio latent info",
    "MARMediaOverlappingConcatenator": "Concatenate media",
    "LatentAVMaskedExtender": "Extend video+audio latent",
    "NoneLatent": "None latent",
    "SaveAVLatent": "Save video+audio latent",
    "LoadAVLatent": "Load video+audio latent",
    "LatentOverlappingConcatenator": "Concatenate video+audio latents",
    "LatentFolderOverlappingConcatenator": "Concatenate video+audio latents from folder",
    "LatentAVCombiner": "Combine video+audio latent",
    "LatentAVSplitter": "Split video+audio latent",
    "LatentAVContrast": "Adjust video latent contrast",
    "VideoLatentPicker": "Pick latent from video"
}