import os,json
import pandas as pd
from PIL import Image
import torch
import open_clip
import torch.nn.functional as F
from sphinx.ext.viewcode import OUTPUT_DIRNAME
from tqdm import tqdm
from pathlib import Path# Aggiungo il percorso della cartella `token-opt` al sys.path così è possibile importare i suoi pacchetti (es. `titok`).
import sys
import torchvision.transforms as T
from tto.test_time_opt import TestTimeOpt, TestTimeOptConfig,CLIPObjective


class ClipOptTester:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.tto_config=None


    def load_rgb_image(self,path:Path):
        image = Image.open(path).convert("RGB")
        return image

    def clip_score_image_text(self,img:Image.Image,text:str) -> float:
        with torch.no_grad():
            image_input=self.clip_preprocess(img).unsqueeze(0).to(self.device)
            text_input=self.clip_tokenizer([text]).to(self.device)

            image_feature=self.clip_model.encode_image(image_input)
            text_feature=self.clip_model.encode_text(text_input)

            image_feature=F.normalize(image_feature,dim=-1)
            text_feature=F.normalize(text_feature,dim=-1)

            sim=(image_feature@text_feature.T).squeeze().item()
        return float(sim)

    def pil_to_tensor(self,seed_img:Image.Image)->torch.Tensor:
        _pre_tto=T.Compose([
            T.Resize(256,interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(256),
            T.ToTensor(),
        ])

        x=_pre_tto(seed_img.convert("RGB")) #[3,H,W] in con valori [0,1]
        x=x.unsqueeze(0).to(self.device,dtype=torch.float32) #[1,3,H,W]
        return x

    def tensor_to_pil(self,tensor_img:torch.Tensor)->Image.Image:
        if tensor_img.ndim==4:
            tensor_img=tensor_img[0]
        tensor_img=tensor_img.detach().clamp(0,1).cpu()
        tensor_img=(tensor_img*255).to(torch.uint8)
        tensor_img=tensor_img.permute(1,2,0).numpy()
        img=Image.fromarray(tensor_img)
        return img

    def set_up_tto_config(self,tto_config:TestTimeOptConfig):
        # Salva la configurazione TTO e, opzionalmente, un objective predefinito
        self.tto_config=tto_config

    def generate_image(self,seed_img: Image.Image, prompt: str) -> Image.Image:
        """
        Esegue Token_Opt su una singola immagine+prompt. Se Token-Opt non è importato correttamente si restituisce l'immagine seed.
        """
        if self.tto_config is None:
            print("Nessuna configurazione TTO impostata, restituisco l'immagine seed")
            return seed_img

        # PIL -> Tensor [1,3,H,W] in [0,1]
        seed_tensor = self.pil_to_tensor(seed_img)  # [1,3,256,256]

        # Obiettivo CLIP testuale (argomento corretto: text=)
        objective = CLIPObjective(prompt=prompt)

        # Usa la configurazione salvata nell'istanza
        tto = TestTimeOpt(self.tto_config, objective).to(device=self.device)
        torch.manual_seed(0)
        tensor_out = tto(seed=seed_tensor)

        return self.tensor_to_pil(tensor_out)
    def _resolve_image_path(self,csv_path:Path,rel_or_abs:str) -> Path:
        """
            Risolve percorsi immagine robustamente:
            - se è assoluto: lo usa
            - altrimenti prova relativo alla cartella del CSV
            - altrimenti prova relativo alla CWD
         """
        p = Path(rel_or_abs)
        if p.is_absolute() and p.exists():
            return p
        cand1 = (csv_path.parent / p).resolve()
        if cand1.exists():
            return cand1
        cand2 = (Path.cwd() / p).resolve()
        if cand2.exists():
            return cand2
        return p  # ultimo tentativo (non esiste; useremo check più avanti)

    def test_on_dataset(self, dataset_csv_path: Path, output_dir: Path) -> pd.DataFrame:
        """
        Esegue il testing batch:
        - legge il CSV (id, macrocat, scenario, object, input_image_path, prompt_target)
        - per ogni riga: salva input, genera output, calcola CLIP score in/out, delta
        - salva per-test JSON e un CSV aggregato in output_dir
        - ritorna un DataFrame con i risultati
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(dataset_csv_path)
        required = ["id", "macrocat", "scenario", "object", "input_image_path", "prompt_target"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Mancano colonne nel CSV: {missing}")

        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Testing"):
            test_id = str(row["id"])
            macrocat = str(row["macrocat"])
            scenario = str(row["scenario"])
            obj = str(row["object"])
            prompt = str(row["prompt_target"])

            # Risolvi path immagine
            img_path = self._resolve_image_path(Path(dataset_csv_path), str(row["input_image_path"]))
            if not img_path.exists():
                print(f"[WARN] Immagine non trovata per {test_id}: {img_path}")
                continue

            # Cartella output per questo test (pulizia selettiva per evitare residui)
            out_dir = output_dir / macrocat / obj / scenario / test_id
            if out_dir.exists():
                # cancella solo la dir del test per evitare confusione tra run
                for f in out_dir.glob("*"):
                    try:
                        f.unlink()
                    except IsADirectoryError:
                        import shutil;
                        shutil.rmtree(f)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Carica seed e salva
            seed_img = self.load_rgb_image(img_path)
            seed_img.save(out_dir / "input.jpg")

            # Calcola CLIP score dell'input (baseline)
            clip_in = self.clip_score_image_text(seed_img, prompt)

            # Genera output
            out_img = self.generate_image(seed_img, prompt)
            out_path = out_dir / "output.png"
            out_img.save(out_path)

            # CLIP score dell'output e delta
            clip_out = self.clip_score_image_text(out_img, prompt)
            delta = clip_out - clip_in

            # Salva meta JSON
            meta = {
                "id": test_id,
                "macrocat": macrocat,
                "scenario": scenario,
                "object": obj,
                "prompt": prompt,
                "input_image_path": str(img_path),
                "output_image_path": str(out_path),
                "clip_in": float(clip_in),
                "clip_out": float(clip_out),
                "delta": float(delta),
            }
            with open(out_dir / "clip_scores.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            results.append(meta)

        res_df = pd.DataFrame(results)
        agg_csv = output_dir / "results_clip.csv"
        res_df.to_csv(agg_csv, index=False)
        print(f"[DONE] Risultati aggregati salvati in: {agg_csv}")

        return res_df