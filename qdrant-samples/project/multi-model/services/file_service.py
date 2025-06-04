import os
from typing import List, Tuple, Dict

class FileService:
    """Handle all file operations"""
    
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    
    @staticmethod
    def get_image_files(folder_path: str) -> Tuple[int, List[str]]:
        """Get all image files from directory"""
        if not os.path.exists(folder_path):
            return 0, []
            
        image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(FileService.VALID_EXTENSIONS):
                image_files.append(filename)
        
        image_files.sort()
        return len(image_files), image_files

    @staticmethod
    def get_cat_descriptions() -> Dict[int, str]:
        """Return predefined cat descriptions"""
        descriptions = {
            0: "This is a black cat on wood in house.",
            1: "This is a black cat with orange eyes.",
            2: "This is a black cat with scary mood.",
            3: "This is a black cat on grass in angry mood.",
            4: "This is a gray cat with orange or yellow eyes.",
            5: "This is a white cat in white place.",
            6: "This is a white cat with black eyes.",
            7: "This is a white cat.",
            8: "This is a white cat on pillow.",
            9: "This is a white cat on grass."
        }
        return descriptions
