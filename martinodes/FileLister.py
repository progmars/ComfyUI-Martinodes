import os
import folder_paths
from .shared import CATEGORY


class FileLister:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": ""}),
            },
            "optional" : {
                "refresh_trigger": ("INT", {"default": 0, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("filepaths", "list_length")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Lists files in a folder in ascending filepath order."


    @staticmethod
    def resolve_folder(input_path=""):
        input_path = input_path.strip()
        if os.path.isabs(input_path):
            candidates = [input_path]
        else:
            candidates = [
                os.path.join(folder_paths.get_output_directory(), input_path),
                os.path.join(folder_paths.get_input_directory(), input_path),
            ]

        for candidate in candidates:
            folder = os.path.realpath(candidate)
            if os.path.isdir(folder):
                return folder

        raise FileNotFoundError(f"FileLister: Folder not found: '{input_path}'")


    @classmethod
    def IS_CHANGED(cls, input_path="", refresh_trigger=0):
        if input_path is None:
            return float("NaN") # work around the cache passing None and throwing WARNINGs here.

        try:
            folder = cls.resolve_folder(input_path)
        except FileNotFoundError:
            return f"missing:{input_path.strip()}"

        file_state = []
        for filepath in cls.list_files(folder):
            stat = os.stat(filepath)
            file_state.append((filepath, stat.st_mtime_ns, stat.st_size))

        return str((folder, file_state))


    @staticmethod
    def list_files(folder):
        return sorted(
            os.path.realpath(os.path.join(folder, filename))
            for filename in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, filename))
        )


    def run(self, input_path="", refresh_trigger=0):
        try:
            folder = self.resolve_folder(input_path)
        except FileNotFoundError:
            return ([], 0)

        filepaths = self.list_files(folder)
        return (filepaths, len(filepaths))
