from .martinodes.MediaSlicer import MediaSlicer
from .martinodes.AudioResampler import AudioResampler
from .martinodes.AudioDurationMatcher import AudioDurationMatcher
from .martinodes.AudioPadder import AudioPadder
from .martinodes.AudioInfo import AudioInfo
from .martinodes.AudioLoader import AudioLoader
from .martinodes.AudioSilenceSplitter import AudioSilenceSplitter
from .martinodes.AudioListSlicer import AudioListSlicer
from .martinodes.LatentAVInfo import LatentAVInfo
from .martinodes.MediaOverlappingConcatenator import MediaOverlappingConcatenator
from .martinodes.LatentAVMaskedExtender import LatentAVMaskedExtender
from .martinodes.LatentAVLoadSave import NonePlugger, SaveAVLatent, LoadAVLatent
from .martinodes.LatentOverlappingConcatenator import LatentOverlappingConcatenator, LatentListOverlappingConcatenator
from .martinodes.LatentAVCombiner import LatentAVCombiner
from .martinodes.LatentAVSplitter import LatentAVSplitter
from .martinodes.LatentAVContrast import LatentAVContrast
from .martinodes.VideoLatentPicker import VideoLatentPicker
from .martinodes.LatentAVSlicer import LatentAVSlicer
from .martinodes.MinimaxH3KeyframeConditioningInjector import MinimaxH3KeyframeConditioningInjector, MinimaxH3AnyFrameConditioningInjector
from .martinodes.BatchRandomPicker import BatchRandomPicker
from .martinodes.BatchBoundPicker import BatchBoundPicker
from .martinodes.FileLister import FileLister
from .martinodes.WorkflowLoop import WorkflowLoop

NODE_CLASS_MAPPINGS = {
    "MARMediaSlicer": MediaSlicer,
    "MARAudioResampler": AudioResampler,
    "MARAudioDurationMatcher": AudioDurationMatcher,
    "MARAudioPadder": AudioPadder,
    "MARAudioInfo": AudioInfo,
    "MARAudioLoader": AudioLoader,
    "MARAudioSilenceSplitter": AudioSilenceSplitter,
    "MARAudioListSlicer": AudioListSlicer,
    "MARMediaOverlappingConcatenator": MediaOverlappingConcatenator,
    "MARLatentAVMaskedExtender": LatentAVMaskedExtender,
    "MARLatentAVInfo": LatentAVInfo,
    "MARNonePlugger": NonePlugger,
    "MARSaveAVLatent": SaveAVLatent,
    "MARLoadAVLatent": LoadAVLatent,
    "MARLatentOverlappingConcatenator": LatentOverlappingConcatenator,
    "MARLatentListOverlappingConcatenator": LatentListOverlappingConcatenator,
    "MARLatentAVCombiner": LatentAVCombiner,
    "MARLatentAVSplitter": LatentAVSplitter,
    "MARLatentAVContrast": LatentAVContrast,
    "MARVideoLatentPicker": VideoLatentPicker,
    "MARLatentAVSlicer": LatentAVSlicer,
    "MARMinimaxH3KeyframeConditioningInjector": MinimaxH3KeyframeConditioningInjector,
    "MARMinimaxH3AnyFrameConditioningInjector": MinimaxH3AnyFrameConditioningInjector,
    "MARBatchRandomPicker": BatchRandomPicker,
    "MARBatchBoundPicker": BatchBoundPicker,
    "MARFileLister": FileLister,
    "MARWorkflowLoop": WorkflowLoop
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MARMediaSlicer": "Slice media",
    "MARAudioInfo": "Audio info",
    "MARAudioLoader": "Load audio from path",
    "MARAudioSilenceSplitter": "Split audio at silences",
    "MARAudioListSlicer": "Slice audio into list",
    "MARAudioResampler": "Resample audio",
    "MARAudioDurationMatcher": "Ensure audio duration",
    "MARAudioPadder": "Pad audio",
    "MARLatentAVInfo": "Video+audio latent info",
    "MARMediaOverlappingConcatenator": "Concatenate media",
    "MARLatentAVMaskedExtender": "Extend video+audio latent",
    "MARNonePlugger": "None plug",
    "MARSaveAVLatent": "Save video+audio latent",
    "MARLoadAVLatent": "Load video+audio latent",
    "MARLatentOverlappingConcatenator": "Concatenate video+audio latents",
    "MARLatentListOverlappingConcatenator": "Concatenate video+audio latent list",
    "MARLatentAVCombiner": "Combine video+audio latent",
    "MARLatentAVSplitter": "Split video+audio latent",
    "MARLatentAVContrast": "Adjust video latent contrast",
    "MARVideoLatentPicker": "Pick latent from video",
    "MARLatentAVSlicer": "Slice video+audio latent",
    "MARMinimaxH3KeyframeConditioningInjector": "Inject first or last frame into H3 conditioning",
    "MARMinimaxH3AnyFrameConditioningInjector": "Inject any frame into H3 conditioning",
    "MARBatchRandomPicker": "Pick random batch item",
    "MARBatchBoundPicker": "Pick first or last batch item",
    "MARFileLister": "List files in folder",
    "MARWorkflowLoop": "Retrigger workflow"
}