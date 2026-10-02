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


def is_safe_path(target_path):
    target = os.path.abspath(target_path)
    allowed_dirs = [
        folder_paths.get_output_directory(),
        folder_paths.get_temp_directory(),
        folder_paths.get_input_directory(),
    ]
    for d in allowed_dirs:
        if not d:
            continue
        base = os.path.abspath(d)
        try:
            if os.path.commonpath([base, target]) == base:
                return True
        except ValueError:
            continue
    return False


def resolve_file_path(path_str):
    if not path_str:
        return None
    output_dir = folder_paths.get_output_directory()
    if os.path.isabs(path_str):
        full_path = os.path.abspath(path_str)
    else:
        full_path = os.path.abspath(os.path.join(output_dir, path_str))
    if is_safe_path(full_path):
        return full_path
    return None


@PromptServer.instance.routes.get("/martinodes/videos")
async def list_videos(request):
    video_regex = request.rel_url.query.get("regex", DEFAULT_VIDEO_REGEX)
    try:
        videos = find_videos(video_regex)
    except re.error as e:
        return web.json_response({"error": f"Invalid regex: {e}", "videos": []}, status=400)
    return web.json_response({"videos": videos})


@PromptServer.instance.routes.post("/martinodes/delete_video_and_latent")
async def delete_video_and_latent(request):
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON payload: {e}"}, status=400)

    video_path = data.get("video_path")
    if not video_path:
        return web.json_response({"error": "Missing 'video_path' in request"}, status=400)

    video_regex = data.get("video_regex", DEFAULT_VIDEO_REGEX)
    latent_pattern = data.get("latent_pattern", DEFAULT_LATENT_PATTERN)
    latent_path = data.get("latent_path")

    if not latent_path:
        try:
            latent_path = build_latent_path(video_path, video_regex, latent_pattern)
        except Exception:
            latent_path = ""

    video_full = resolve_file_path(video_path)
    if not video_full:
        return web.json_response({"error": f"Invalid or disallowed video path: {video_path}"}, status=403)

    latent_full = resolve_file_path(latent_path) if latent_path else None

    video_deleted = False
    latent_deleted = False

    if os.path.isfile(video_full):
        try:
            os.remove(video_full)
            video_deleted = True
        except OSError as e:
            return web.json_response({"error": f"Failed to delete video file: {e}"}, status=500)
    elif os.path.exists(video_full):
        return web.json_response({"error": f"Target video is not a regular file: {video_path}"}, status=400)

    if latent_full and os.path.isfile(latent_full):
        try:
            os.remove(latent_full)
            latent_deleted = True
        except OSError as e:
            return web.json_response({
                "error": f"Video deleted, but failed to delete latent file: {e}",
                "video_deleted": video_deleted,
                "latent_path": latent_path,
            }, status=500)

    if not video_deleted and not latent_deleted:
        if not os.path.exists(video_full):
            return web.json_response({"error": f"Video file not found: {video_path}"}, status=404)

    return web.json_response({
        "success": True,
        "video_deleted": video_deleted,
        "latent_deleted": latent_deleted,
        "video_path": video_path,
        "latent_path": latent_path if (latent_full and latent_deleted) else "",
    })


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
