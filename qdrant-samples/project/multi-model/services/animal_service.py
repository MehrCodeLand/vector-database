from models.animal import Animal
from services.file_service import FileService
from typing import List, Optional

class AnimalService:
    """Handle animal data operations"""
    
    def __init__(self, images_directory: str = "images/cats"):
        self.images_directory = images_directory
        self.file_service = FileService()

    def create_animals_from_directory(self) -> List[Animal]:
        """Create animal objects from images in directory"""
        animals = []
        
        try:
            count, filenames = self.file_service.get_image_files(self.images_directory)
            descriptions = self.file_service.get_cat_descriptions()
            
            for i, filename in enumerate(filenames):
                # Use modulo to cycle through descriptions if we have more images than descriptions
                desc_index = i % len(descriptions)
                description = descriptions.get(desc_index, "A cat image")
                
                animal = Animal(
                    name=filename,
                    category="animal",
                    description=description,
                    type_of_category="cat",
                    image_path=f"{self.images_directory}/{filename}"
                )
                animals.append(animal)
                
            return animals
        except Exception as e:
            print(f"Error creating animals from directory: {e}")
            return []