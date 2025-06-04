from dataclasses import dataclass
from typing import Optional

@dataclass
class Animal:
    """Animal data model with proper type hints"""
    name: str
    category: str
    description: str
    type_of_category: str
    image_path: Optional[str] = None

    def to_dict(self):
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "type_of_category": self.type_of_category,
            "image_path": self.image_path
        }
