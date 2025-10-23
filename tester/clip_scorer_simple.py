# clip_scorer_simple.py
from pathlib import Path
from typing import List, Union, Iterable, Tuple
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

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
