import os
import shutil
import folder_paths
from .shared import CATEGORY


class FolderDeleter:
    """
    Deletes a folder and its contents. Useful for API automation workflows
    to clean up temporary files and directories.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Deletes a folder and all its contents. Returns status message."

    @staticmethod
    def resolve_folder(folder_path=""):
        """Resolve folder path, supporting relative and absolute paths."""
        folder_path = folder_path.strip()
        
        if not folder_path:
            raise ValueError("FolderDeleter: Folder path cannot be empty")
        
        if os.path.isabs(folder_path):
            candidates = [folder_path]
        else:
            candidates = [
                os.path.join(folder_paths.get_output_directory(), folder_path),
                os.path.join(folder_paths.get_input_directory(), folder_path),
            ]

        for candidate in candidates:
            folder = os.path.realpath(candidate)
            if os.path.isdir(folder):
                return folder

        raise FileNotFoundError(f"FolderDeleter: Folder not found: '{folder_path}'")


    def run(self, folder_path=""):
        try:
            folder = self.resolve_folder(folder_path)
        except (ValueError, FileNotFoundError) as e:
            error_msg = str(e)
            print(error_msg)
            return (error_msg,)

        try:
            shutil.rmtree(folder)
            status = f"Successfully deleted folder: {folder}"
            print(status)
            return (status,)
        except Exception as e:
            error_msg = f"FolderDeleter: Failed to delete folder '{folder}'. Error: {e}"
            print(error_msg)
            return (error_msg,)
