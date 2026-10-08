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
from .martinodes.LatentAVSlicer import LatentAVSlicer
from .martinodes.MinimaxH3KeyframeConditioningInjector import MinimaxH3KeyframeConditioningInjector

NODE_CLASS_MAPPINGS = {
    "MARMediaSlicer": MediaSlicer,
    "MARAudioResampler": AudioResampler,
    "MARAudioTrimExtender": AudioTrimExtender,
    "MARMediaTrimmer": MediaTrimmer,
    "MARAudioInfo": AudioInfo,
    "MARMediaOverlappingConcatenator": MediaOverlappingConcatenator,
    "MARLatentAVMaskedExtender": LatentAVMaskedExtender,
    "MARLatentAVInfo": LatentAVInfo,
    "MARNoneLatent": NoneLatent,
    "MARSaveAVLatent": SaveAVLatent,
    "MARLoadAVLatent": LoadAVLatent,
    "MARLatentOverlappingConcatenator": LatentOverlappingConcatenator,
    "MARLatentFolderOverlappingConcatenator": LatentFolderOverlappingConcatenator,
    "MARLatentAVCombiner": LatentAVCombiner,
    "MARLatentAVSplitter": LatentAVSplitter,
    "MARLatentAVContrast": LatentAVContrast,
    "MARVideoLatentPicker": VideoLatentPicker,
    "MARLatentAVSlicer": LatentAVSlicer,
    "MARMinimaxH3KeyframeConditioningInjector": MinimaxH3KeyframeConditioningInjector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MARMediaSlicer": "Media head or tail",
    "MARAudioInfo": "Audio info",
    "MARAudioResampler": "Resample audio",
    "MARMediaTrimmer": "Trim media",
    "MARAudioTrimExtender": "Ensure audio duration",
    "MARLatentAVInfo": "Video+audio latent info",
    "MARMediaOverlappingConcatenator": "Concatenate media",
    "MARLatentAVMaskedExtender": "Extend video+audio latent",
    "MARNoneLatent": "None latent",
    "MARSaveAVLatent": "Save video+audio latent",
    "MARLoadAVLatent": "Load video+audio latent",
    "MARLatentOverlappingConcatenator": "Concatenate video+audio latents",
    "MARLatentFolderOverlappingConcatenator": "Concatenate video+audio latents from folder",
    "MARLatentAVCombiner": "Combine video+audio latent",
    "MARLatentAVSplitter": "Split video+audio latent",
    "MARLatentAVContrast": "Adjust video latent contrast",
    "MARVideoLatentPicker": "Pick latent from video",
    "MARLatentAVSlicer": "Slice video+audio latent",
    "MARMinimaxH3KeyframeConditioningInjector": "Inject FL frames into H3 conditioning"
}