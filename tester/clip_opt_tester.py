# testing_core.py
from __future__ import annotations
import os, json, math, csv, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
from typing import List, Union, Iterable, Tuple
from PanelSaver import PanelSaver
from PromptConfiguration import PromptConfigurator
from PIL import Image, ImageDraw, ImageFont
import torch
import open_clip

# --- Scorers (tutti offline). CLIPScore è opzionale ---
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

class ClipScorer:
    """
    CLIPScore semplice:
      - model_name: es. "ViT-B-32"
      - pretrained: es. "laion2b_s34b_b79k" (no checkpoint locale necessario)
    Ritorna score in scala [0, 100] come nel paper CLIPScore.
    """
    def __init__(self,
                 model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def score_image_text(self, image: Image.Image, text: str) -> float:
        """
        CLIPScore(I, C) = max(100 * cos(E_I, E_C), 0)
        """
        img_feat = self._encode_images([image])
        txt_feat = self._encode_text([text])
        cos = (img_feat @ txt_feat.T).squeeze().item()
        return max(100.0 * cos, 0.0)

    def score_batch(
        self,
        images: List[Image.Image],
        texts: List[str],
        batch_size: int = 32
    ) -> List[float]:
        assert len(images) == len(texts), "images e texts devono avere stessa lunghezza"
        scores: List[float] = []
        for i in range(0, len(images), batch_size):
            b_imgs = images[i:i+batch_size]
            b_txts = texts[i:i+batch_size]
            with torch.no_grad():
                img_feat = self._encode_images(b_imgs)
                txt_feat = self._encode_text(b_txts)
                cos = (img_feat @ txt_feat.T).diag()
                b_scores = [max(100.0 * v.item(), 0.0) for v in cos]
                scores.extend(b_scores)
        return scores

def score_folder_to_csv(
    images_dir: Union[str, Path],
    prompts: Union[List[str], Iterable[str]],
    out_csv: Union[str, Path],
    scorer: ClipScorer,
):
    """
    Associa ogni immagine ad un prompt (1-1) nell'ordine alfabetico delle immagini.
    Salva CSV con colonne: filename, prompt, clip_score
    """
    import csv
    p = Path(images_dir)
    img_paths = sorted(
        [x for x in p.rglob("*") if x.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )
    prompts = list(prompts)
    assert len(img_paths) == len(prompts), "Numero immagini e prompt deve combaciare"

    images = [Image.open(x).convert("RGB") for x in img_paths]
    scores = scorer.score_batch(images, prompts, batch_size=32)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prompt", "clip_score"])
        for path, prompt, s in zip(img_paths, prompts, scores):
            w.writerow([str(path.relative_to(p)), prompt, f"{s:.3f}"])
    return out_csv

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