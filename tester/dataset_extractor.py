
from __future__ import annotations

import csv
import io
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

# Optional imports (available in Colab)
# - "datasets" for HuggingFace Datasets (ABO)
# - "torchvision" for COCO if used via CocoDetection
# These are imported lazily inside methods to keep this module import-safe
# even when the packages are not installed.


@dataclass
class DatasetConfig:
    base_dir: Path                 # cartella dove salvare i file (es. /content/dataset)
    abo_enabled: bool = True       # se vuoi usare il dataset ABO per immagini "clean"
    coco_images_dir: Optional[Path] = None      # cartella COCO per immagini "real"
    coco_annotations_file: Optional[Path] = None# file JSON di annotazioni COCO

    def ensure_dirs(self) -> None:
        (self.base_dir / "cambio_materiale" / "clean").mkdir(parents=True, exist_ok=True)
        (self.base_dir / "cambio_materiale" / "real").mkdir(parents=True, exist_ok=True)


@dataclass
class DatabaseExtractor:
    """
    Utility for sampling images from public datasets to build a testing CSV
    for generative modifications (material/color/style/add_component).
    This does NOT "create a dataset"; it just reuses existing datasets.
    """
    config: DatasetConfig
    objects_en: List[str] = field(default_factory=lambda: ["cup", "chair", "table", "vase", "lamp", "sofa"])

    # Mapping EN object -> ABO search keyword(s)
    abo_query_map: Dict[str, str] = field(default_factory=lambda: {
        "cup": "cup",
        "chair": "chair",
        "table": "table",
        "vase": "vase",
        "lamp": "lamp",
        "sofa": "sofa",
    })

    # Prompt templates and value banks
    prompt_templates: Dict[str, List[str]] = field(default_factory=lambda: {
        "change_material": [
            "transform the {obj} into a {mat} {obj}, while keeping the original proportions and realistic lighting."
        ],
        "change_color": [
            "change the {obj} color to {color}, while keeping the original proportions and realistic lighting."
        ],
        "change_style": [
            "convert the {obj} into a {style} design {obj}, while keeping the original proportions and realistic lighting."
        ],
        "add_component": [
            "add a {component} to the {obj}, while keeping the original proportions and realistic lighting."
        ],
    })

    materials: List[str] = field(default_factory=lambda: ["polished metal", "transparent glass", "glossy plastic", "polished marble", "matte ceramic", "natural stone"
    ])
    colors: List[str] = field(default_factory=lambda: ["red", "blue", "black", "white", "green"])
    styles: List[str] = field(default_factory=lambda: ["minimalist", "industrial", "futuristic", "vintage"])
    components: List[str] = field(default_factory=lambda: ["handle", "shelf", "armrest", "shade", "base"])

    # Internal stores for sampled images
    clean_index: Dict[str, List[Path]] = field(default_factory=dict)  # obj_en -> list of paths
    real_index: Dict[str, List[Path]] = field(default_factory=dict)   # obj_en -> list of paths

    def _save_image(self, img: Image.Image, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out_path)
        return out_path

    # --------------------------
    # CLEAN images from ABO
    # --------------------------
    def fetch_clean_from_abo(self, obj_en: str, k: int = 3) -> List[Path]:
        """
        Sample k product images for the given object from ABO (HuggingFace).
        Requires internet access. Saves under base_dir/cambio_materiale/clean/.
        """
        if not self.config.abo_enabled:
            return []

        try:
            from datasets import load_dataset
            import requests
        except Exception as e:
            print("[ABO] Missing deps (datasets/requests):", e)
            return []

        abo = load_dataset("amazon-berkeley-objects", split="train")
        key = (self.abo_query_map.get(obj_en, obj_en) or "").lower()

        # Filter ABO by product_type/title match (simple heuristic)
        def _flt(x):
            pt = (x.get("product_type") or "").lower()
            title = (x.get("title") or "").lower()
            return (key in pt) or (key in title)

        saved: List[Path] = []
        base_out = self.config.base_dir / "cambio_materiale" / "clean"
        for ex in abo:
            if not _flt(ex):
                continue
            url = ex.get("image_url") or ex.get("main_image_url")
            if not url:
                continue
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                out_path = base_out / f"{obj_en}_{len(saved):02d}.jpg"
                self._save_image(img, out_path)
                saved.append(out_path)
                if len(saved) >= k:
                    break
            except Exception:
                continue

        self.clean_index.setdefault(obj_en, []).extend(saved)
        return saved

    # --------------------------
    # REAL images from COCO
    # --------------------------
    def fetch_real_from_coco(self, obj_en: str, k: int = 3) -> List[Path]:
        """
        Sample k real-world images for the given object from COCO.
        Requires local COCO images and annotations (set in config).
        Saves under base_dir/cambio_materiale/real/.
        """
        if self.config.coco_images_dir is None or self.config.coco_annotations_file is None:
            print("[COCO] Provide coco_images_dir and coco_annotations_file in DatasetConfig.")
            return []

        from torchvision.datasets import CocoDetection

        coco = CocoDetection(root=str(self.config.coco_images_dir),
                             annFile=str(self.config.coco_annotations_file))

        # Load categories map
        with open(self.config.coco_annotations_file, "r") as f:
            cats = {c["id"]: c["name"] for c in json.load(f)["categories"]}

        targets_by_obj = {
            "cup": {"cup"},
            "chair": {"chair"},
            "table": {"dining table", "table"},
            "vase": {"vase"},
            "lamp": {"lamp"},
            "sofa": {"couch", "sofa"},
        }
        wanted = targets_by_obj.get(obj_en, {obj_en})

        saved: List[Path] = []
        base_out = self.config.base_dir / "cambio_materiale" / "real"

        for img, anns in coco:
            cat_names = {cats.get(a["category_id"], "") for a in anns}
            if wanted & cat_names:
                out_path = base_out / f"{obj_en}_{len(saved):02d}.jpg"
                img.convert("RGB").save(out_path)
                saved.append(out_path)
                if len(saved) >= k:
                    break

        self.real_index.setdefault(obj_en, []).extend(saved)
        return saved

    # --------------------------
    # Prompt factory
    # --------------------------
    def make_prompt(self, category: str, obj_en: str) -> str:
        tmpls = self.prompt_templates[category]
        tmpl = random.choice(tmpls)
        if category == "change_material":
            return tmpl.format(obj=obj_en, mat=random.choice(self.materials))
        if category == "change_color":
            return tmpl.format(obj=obj_en, color=random.choice(self.colors))
        if category == "change_style":
            return tmpl.format(obj=obj_en, style=random.choice(self.styles))
        if category == "add_component":
            return tmpl.format(obj=obj_en, component=random.choice(self.components))
        return tmpl.format(obj=obj_en)

    # --------------------------
    # Build CSV
    # --------------------------
    def build_csv(self,
                  out_csv: Path,
                  categories: List[str],
                  clean_per_obj: int = 3,
                  real_per_obj: int = 3,
                  objects: Optional[List[str]] = None) -> Path:
        """
        Build a CSV that your testing pipeline can read directly.
        It will sample images from ABO/COCO and write rows for each (category × image).
        CSV columns: id, macrocat, scenario, object, input_image_path, prompt_target
        """
        self.config.ensure_dirs()
        objs = objects or self.objects_en

        # 1) Sample images
        if self.config.abo_enabled:
            for obj in objs:
                if clean_per_obj > 0:
                    self.fetch_clean_from_abo(obj, k=clean_per_obj)

        if self.config.coco_images_dir and self.config.coco_annotations_file:
            for obj in objs:
                if real_per_obj > 0:
                    self.fetch_real_from_coco(obj, k=real_per_obj)

        # 2) Build rows
        rows: List[dict] = []
        rid = 1

        def add_rows_for(paths: List[Path], scenario: str, obj: str):
            nonlocal rid, rows
            for p in paths:
                for cat in categories:
                    rows.append({
                        "id": f"{cat}_{obj}_{scenario}_{rid:05d}",
                        "macrocat": cat,
                        "scenario": scenario,
                        "object": obj,
                        "input_image_path": str(p),
                        "prompt_target": self.make_prompt(cat, obj),
                    })
                    rid += 1

        for obj in objs:
            for p in self.clean_index.get(obj, []):
                add_rows_for([p], "clean", obj)
            for p in self.real_index.get(obj, []):
                add_rows_for([p], "real", obj)

        # 3) Write CSV
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id","macrocat","scenario","object","input_image_path","prompt_target"])
            writer.writeheader()
            writer.writerows(rows)

        return out_csv
