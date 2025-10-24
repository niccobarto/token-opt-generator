# testing_core.py
from __future__ import annotations
import os, json, math, csv, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from ImageSaver import PanelSaver
from PromptConfigurator import PromptConfigurator
from PIL import Image, ImageDraw, ImageFont
from clip_scorer_simple import ClipScorer

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


# ============= Config & Session (solo CLIP) =============
@dataclass
class TestConfig:
    inputs_root: Path
    outputs_root: Path
    split: str                 # "clean" | "real"
    object_name: str           # es. "chair"
    category: str              # "change_material", "change_color", "change_style", "add_component"
    n_prompts_per_image: int = 3
    seed: Optional[int] = 0
    # mapping opzionale [-1,1] -> [0,100] per leggibilità
    map_clip_to_0_100: bool = False

class TestSession:
    def __init__(
        self,
        cfg: TestConfig,                           # Config di esecuzione (cartelle, split, oggetto, categoria, seed, ecc.)
        prompt_cfg: PromptConfigurator,            # Generatore dei prompt in base a categoria/oggetto
        edit_fn: Callable[[Image.Image, str], Image.Image],  # Funzione “adapter” che esegue l’editing (immagine + prompt -> immagine)
        clip_model: Any                            # Modello CLIP (già caricato) da usare per lo scoring
    ):
        self.cfg = cfg                             # Salvo la config
        self.prompt_cfg = prompt_cfg               # Salvo il configuratore dei prompt
        self.edit_fn = edit_fn                     # Salvo la funzione di editing
        self.panel_saver = PanelSaver()            # Istanzio utility per creare il pannello (originale|edit + prompt)
        self.scorer = ClipScorer(clip_model)        # Inizializzo lo scorer CLIP (cosine similarity testo-immagine)

        # I/O
        self.input_dir = Path(cfg.inputs_root) / cfg.split / cfg.object_name            # Directory sorgente immagini (inputs/<split>/<oggetto>)
        self.output_dir = Path(cfg.outputs_root) / cfg.split / cfg.object_name / cfg.category  # Directory di output (outputs/<split>/<oggetto>/<categoria>)
        self.summary_csv = Path(cfg.outputs_root) / cfg.split / cfg.object_name / f"summary_{cfg.category}.csv"  # CSV riassuntivo per categoria

    def _list_images(self) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # Estensioni supportate
        if not self.input_dir.exists():                    # Se la cartella non esiste
            return []                                      # Non ci sono immagini
        return sorted([p for p in self.input_dir.iterdir() if p.suffix.lower() in exts])  # Lista ordinata dei file immagine

    def _write_summary_header(self):
        self.summary_csv.parent.mkdir(parents=True, exist_ok=True)  # Assicuro che la dir del CSV esista
        if not self.summary_csv.exists():                           # Se il CSV non c’è ancora
            with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:  # Lo creo in scrittura
                writer = csv.writer(f)
                # Intestazione del CSV (colonne): nota overall=clip e mettiamo anche il seed usato
                writer.writerow(["image_name", "prompt", "clip", "overall", "panel_path", "seed_used"])

    def _append_summary_row(self, image_name: str, prompt: str, clip_score: float, panel_path: Path, seed_used: int):
        score = clip_score                                           # Score di base = CLIP cosine similarity in [-1, 1]
        if self.cfg.map_clip_to_0_100:                               # Se richiesto, rimappo in [0, 100]
            # mappa [-1,1] -> [0,100] linearmente:  -1 -> 0, 0 -> 50, 1 -> 100
            score = 50.0 * (clip_score + 1.0)
        with open(self.summary_csv, "a", newline="", encoding="utf-8") as f:  # Apro il CSV in append
            writer = csv.writer(f)
            # Scrivo la riga: image file, prompt, clip (già mappato se attivo), overall (uguale a clip), path pannello, seed locale
            writer.writerow([image_name, prompt, score, score, str(panel_path), seed_used])

    def run(self) -> None:
        self._write_summary_header()                 # Prepara il CSV (crea header se serve)
        rng = random.Random(self.cfg.seed)           # RNG deterministico dal seed principale per riproducibilità

        images = self._list_images()                 # Colleziono le immagini da processare
        if not images:                               # Se non ne trovo, warno e finisco
            print(f"[WARN] Nessuna immagine trovata in: {self.input_dir}")
            return

        print(f"[INFO] Found {len(images)} images in {self.input_dir}")  # Log informativo

        for img_path in images:                      # Loop su ciascun file immagine
            try:
                orig = Image.open(img_path).convert("RGB")  # Apro l’immagine e forzo RGB per coerenza
            except Exception as e:
                print(f"[SKIP] {img_path.name}: {e}")       # Se fallisce l’apertura, salto con messaggio
                continue

            # Seed secondario per questa immagine (deterministico dato cfg.seed): lo salvo anche nel CSV
            local_seed = rng.randint(0, 10_000)

            prompts = self.prompt_cfg.make_prompts(  # Genero i prompt per questa immagine
                object_name=self.cfg.object_name,    # Oggetto (es. "chair")
                category=self.cfg.category,          # Categoria (es. "change_material")
                n=self.cfg.n_prompts_per_image,      # Quanti prompt vuoi per immagine
                seed=local_seed,                     # Seed locale (così i prompt sono ripetibili per questa immagine)
            )

            for pr in prompts:                       # Loop sui prompt generati
                # 1) Editing: chiami la tua pipeline (Token-Opt adapter) che produce l’immagine modificata
                edited = self.edit_fn(orig, pr)

                # 2) Scoring CLIP: similarità testo (prompt) vs immagine EDITED (cosine in [-1, 1])
                clip_val = self.scorer.score(edited, pr)

                # 3) Costruzione del nome file output:
                #    pr.split()[-1] prende l'ultima parola del prompt come "chiave" rapida per leggibilità
                key = pr.split()[-1].replace("/", "_")[:20]  # Sanitizza e tronca per sicurezza su FS
                panel_name = f"{Path(img_path).stem}__{key}.png"  # <nomeInput>__<chiave>.png
                out_path = self.output_dir / panel_name            # Path completo di output

                # 4) Creo e salvo il pannello affiancato [originale | edited] con prompt in basso
                self.panel_saver.make_panel(orig, edited, pr, out_path)

                # 5) Accodo una riga nel CSV con punteggio e metadati
                self._append_summary_row(Path(img_path).name, pr, clip_val, out_path, local_seed)

        print(f"[DONE] Summary scritto in: {self.summary_csv}")  # Log finale