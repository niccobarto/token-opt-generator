import os
from pathlib import Path
from typing import List
from PIL import Image

#Devo restituire una lista di oggetti del tipo ("oggetto", list immagini clean, list immagini real)
class Extractor:

    def __init__(self,  input_root: Path):
        self.input_root = input_root

    def extract_paths(self,obj:str) -> tuple[str, List[Path], List[Path]]:
        clean_dir=self.input_root /"clean" / obj
        real_dir=self.input_root /"real" / obj

        clean_list = []
        for p in clean_dir.iterdir():
            clean_list.append(Path(clean_dir / p.name))

        real_list = []
        for p in real_dir.iterdir():
            real_list.append(Path(real_dir / p.name))
        return obj, clean_list, real_list

    def extract_images(self,obj:str, clean_paths:List[Path], real_paths:List[Path]) -> tuple[str, List[Image.Image], List[Image.Image]]:
        clean_images = []
        for p in clean_paths:
            img = Image.open(p).convert("RGB")
            clean_images.append(img)

        real_images = []
        for p in real_paths:
            img = Image.open(p).convert("RGB")
            real_images.append(img)
        return obj, clean_images, real_images

