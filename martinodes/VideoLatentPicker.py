import os
import re
import folder_paths
from aiohttp import web
from server import PromptServer
from .shared import CATEGORY

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".gif")
DEFAULT_VIDEO_REGEX = r"video/.*ltnt_([0-9]+)_.*"
DEFAULT_LATENT_PATTERN = "av_latents/latent_[0-9]+.avlatent"
NUMBER_TOKEN = re.compile(r"\[0-9\]\+?")


def extract_number(match):
    return match.group(1) if match.re.groups else match.group(0)


def find_videos(video_regex):
    output_dir = folder_paths.get_output_directory()
    regex = re.compile(video_regex)
    videos = []
    for root, _, filenames in os.walk(output_dir):
        for f in filenames:
            if not f.lower().endswith(VIDEO_EXTENSIONS):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, output_dir).replace("\\", "/")
            match = regex.match(rel_path)
            if not match:
                continue
            subfolder, filename = os.path.split(rel_path)
            videos.append({
                "path": rel_path,
                "filename": filename,
                "subfolder": subfolder,
                "number": extract_number(match),
                "mtime": os.path.getmtime(full_path),
            })
    videos.sort(key=lambda v: v["mtime"], reverse=True)
    return videos


def build_latent_path(video_path, video_regex, latent_pattern):
    match = re.match(video_regex, video_path)
    if not match:
        return ""
    return NUMBER_TOKEN.sub(lambda _: extract_number(match), latent_pattern, count=1)


@PromptServer.instance.routes.get("/martinodes/videos")
async def list_videos(request):
    video_regex = request.rel_url.query.get("regex", DEFAULT_VIDEO_REGEX)
    try:
        videos = find_videos(video_regex)
    except re.error as e:
        return web.json_response({"error": f"Invalid regex: {e}", "videos": []}, status=400)
    return web.json_response({"videos": videos})


class VideoLatentPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "video_regex": ("STRING", {"default": DEFAULT_VIDEO_REGEX}),
            "latent_pattern": ("STRING", {"default": DEFAULT_LATENT_PATTERN}),
            "selected_video": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("latent_path", )
    FUNCTION = "pick"
    CATEGORY = CATEGORY

    def pick(self, video_regex, latent_pattern, selected_video):
        latent_path = build_latent_path(selected_video, video_regex, latent_pattern) if selected_video else ""
        if selected_video and not latent_path:
            print(f"VideoLatentPicker: '{selected_video}' does not match regex '{video_regex}'.")
        return (latent_path, )
