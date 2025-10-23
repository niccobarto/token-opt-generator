# Test-time token optimization su tokenizer TiTok / VQGAN
# -------------------------------------------------------
# Questo modulo implementa un ciclo di ottimizzazione a test time che opera
# direttamente nello spazio dei token latenti di un tokenizer (TiTok o MaskGIT-VQGAN),
# al fine di massimizzare un obiettivo percettivo (es. similarità con un prompt CLIP/SigLIP).
# L'ottimizzazione aggiorna i token per gradiente, decodifica l'immagine e valuta la loss.
# Opzionalmente usa EMA sui token, mixed precision (AMP), rumore programmato e regolarizzazioni.

from tto.ema import EMAModel
from tto.siglip import SigLIP
from tto.vqgan_wrapper import PretrainedVQGAN

from typing import cast, Callable, Literal
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from einops import rearrange, einsum
from jaxtyping import Float
import open_clip

from titok.modeling.quantizer import DiagonalGaussianDistribution
from titok.modeling.titok import TiTok


@dataclass
class TestTimeOptConfig:
    """Configurazione per l'ottimizzazione a test time.

    Attributi
    - titok_checkpoint: nome o path del checkpoint TiTok oppure "maskgit-vqgan" per usare VQGAN.
    - optimize_post_quantization_tokens: se True ottimizza i token POST quantizzazione (codici discreti
      o media VAE), altrimenti ottimizza i token PRE quantizzazione e li quantizza ad ogni decode.
    - vae_deterministic_sampling: per quantizzazione VAE, usa la media (deterministico) o sampling.
    - lr: learning rate dell'ottimizzatore Adam che aggiorna i token.
    - ema_decay: coefficiente di decay per EMA sui token; 0 disabilita l'effetto.
    - token_noise: ampiezza del rumore gaussiano aggiunto ai token, con scheduling decrescente.
    - reg_weight: peso della regolarizzazione (L2 verso seed o verso zero, vedi reg_type).
    - reg_type: None, "seed" (ancoraggio ai token iniziali) o "zero" (spinge i token a ridursi).
    - num_iter: numero massimo di iterazioni di ottimizzazione.
    - enable_amp: abilita l'autocast fp16 durante forward/loss per accelerare e ridurre memoria.
    """
    titok_checkpoint: str = "yucornetto/tokenizer_titok_l32_imagenet"
    optimize_post_quantization_tokens: bool = False
    vae_deterministic_sampling: bool = True
    lr: float = 1e-1
    ema_decay: float = 0.
    token_noise: float | None = None
    reg_weight: float | None = None
    reg_type: None | Literal["seed", "zero"] = None
    num_iter: int = 600
    enable_amp: bool = False


@dataclass
class TestTimeOptInfo:
    """Pacchetto di info passato a callback e token_reset_callback.

    Campi
    - i: indice iterazione corrente (0-based).
    - tokens: tokens latenti correnti (dopo step e, se abilitato, con EMA applicato durante callback).
    - img: immagine corrente decodificata (senza EMA sul decoder, solo sui token).
    - loss: loss per batch element (valore negativo della similarità quando si usa CLIP/SigLIP).
    """
    i: int
    tokens: Float[Tensor, "b d 1 n"]
    img: Float[Tensor, "b c h w"]
    loss: Float[Tensor, "b"]


# Obiettivo: prende in input immagini (b,c,h,w) e restituisce loss per elemento (b,)
ObjectiveT = Callable[[Float[Tensor, "b c h w"]], Float[Tensor, "b"]]


class TestTimeOpt(nn.Module):
    """Ciclo di ottimizzazione a test time sui token latenti.

    Contratto I/O
    - Input: seed (b,c,h,w) oppure seed_tokens (b,d,1,n) per inizializzare i token.
    - Output: immagine ricostruita finale clampata in [0,1].

    Note
    - Se `optimize_post_quantization_tokens` è False, ottimizziamo nello spazio PRE-quantizzazione e
      applichiamo la quantizzazione ad ogni decode. Questo mantiene coerenza con il codice discreto.
    - Se True, ottimizziamo direttamente i codici POST-quantizzazione (discreti o mean VAE), utile
      se si vuole stabilità e coerenza con il decoder.
    - EMA stabilizza l'ottimizzazione: il callback e l'output finale vengono valutati con i pesi EMA.
    """
    def __init__(
        self,
        config: TestTimeOptConfig,
        objective: ObjectiveT,
    ):
        super().__init__()
        self.config = config
        self.objective = objective
        # Carica tokenizer/decoder: TiTok di default, oppure MaskGIT-VQGAN se richiesto.
        if config.titok_checkpoint == "maskgit-vqgan":
            print("Using pretrained MaskGIT-VQGAN!")
            self.titok = PretrainedVQGAN()
        else:
            self.titok = TiTok.from_pretrained(config.titok_checkpoint)
        self.eval()  # questo modulo opera in eval; solo i token sono "parametri" ottimizzati

    def decode(self, tokens: Float[Tensor, "b d 1 n"]) -> Float[Tensor, "b c h w"]:
        """Decodifica tokens in immagine, gestendo eventualmente la quantizzazione.

        - Se ottimizziamo pre-quantizzazione, applichiamo la quantizzazione prima della decode.
        - Per VAE, scegliamo mean o sample a seconda di `vae_deterministic_sampling`.
        """
        def _maybe_quantize(tokens):
            if self.config.optimize_post_quantization_tokens:
                # Ottimizziamo già nello spazio post-quantizzazione (codici pronti per il decoder)
                return tokens
            else:
                # Ottimizziamo nello spazio pre-quantizzazione: serve mappare a post-quantizzazione
                if self.titok.quantize_mode == "vae":
                    assert isinstance(self.titok, TiTok)
                    tokens = self.titok.quantize(tokens)
                    return (
                        tokens.mean
                        if self.config.vae_deterministic_sampling
                        else tokens.sample()
                    )
                else:
                    # quantize() ritorna (codes, indices, ...). Qui usiamo i codici continui per decode
                    return self.titok.quantize(tokens)[0]  # type: ignore
        tokens = _maybe_quantize(tokens)
        dec = self.titok.decode(tokens)
        return dec

    def encode(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b d 1 n"]:
        """Codifica un'immagine in token latenti.

        Se si ottimizzano i token post-quantizzazione, si applica la mappatura qui per partire
        subito nello spazio corretto (VAE: mean/sample; codebook: indice->codice continuo).
        """
        tok = self.titok.encoder(pixel_values=img, latent_tokens=self.titok.latent_tokens)
        if self.config.optimize_post_quantization_tokens:
            if self.titok.quantize_mode == "vae":
                tok = DiagonalGaussianDistribution(tok)
                return tok.mean if self.config.vae_deterministic_sampling else tok.sample()
            else:
                assert isinstance(tok, Tensor)
                return self.titok.quantize(tok)[0]  # type: ignore
        return tok

    def _token_noise_schedule(self, i):
        """Scheduler di decadimento del rumore (andamento cos^2):
        - i: indice iterazione
        - ritorna fattore in [0,1] che parte alto e tende a 0 verso ~2/3 delle iterazioni.
        Utile per esplorare all'inizio e stabilizzare verso la fine.
        """
        t = i / (self.config.num_iter - 1)
        t = max(0, min(1, 1.5 * t))  # rampa a 0 ai 2/3 di num_iter circa
        return 0.5 * (1 + np.cos(np.pi * t))

    def forward(
        self,
        seed: Float[Tensor, "b c h w"] | None,
        seed_tokens: Float[Tensor, "b d 1 n"] | None = None,
        callback: Callable[[TestTimeOptInfo], bool | None] | None = None,
        token_reset_callback: Callable[[TestTimeOptInfo], Float[Tensor, "b d 1 n"] | None] | None = None,
    ):
        """Esegue il loop di ottimizzazione.

        Parametri
        - seed: immagine di partenza; se fornita, viene encodata in token iniziali.
        - seed_tokens: tokens iniziali alternativi (mutuamente esclusivo con `seed`).
        - callback(info) -> bool|None: chiamato ad ogni iterazione con info; se ritorna True, interrompe.
        - token_reset_callback(info) -> tokens|None: può restituire nuovi token (es. reset/ri-proiezione).

        Ritorna
        - immagine finale decodificata dai token EMA e clampata in [0,1].
        """
        assert not self.training
        if seed is not None:
            if seed_tokens is not None:
                raise ValueError("must provide seed_tokens or seed but not both")
            # Encodiamo il seed in token iniziali senza tracciare gradiente
            with torch.no_grad():
                opt_tokens = self.encode(seed)
        else:
            if seed_tokens is None:
                raise ValueError("must provide either seed_tokens or seed")
            # Cloniamo i token forniti per poterli ottimizzare in-place
            opt_tokens = seed_tokens.detach().clone()

        # I token sono i "parametri" da ottimizzare
        opt_tokens.requires_grad_(True)
        opt = torch.optim.Adam(params=[opt_tokens], lr=self.config.lr)
        scaler = torch.GradScaler(enabled=self.config.enable_amp)  # per AMP
        # EMA sui token per stabilità e per l'output finale
        ema = EMAModel(
            [opt_tokens],
            decay=self.config.ema_decay,
            min_decay=self.config.ema_decay
        )
        orig_tokens = opt_tokens.detach().clone()  # per regolarizzazione "seed"

        for i in range(self.config.num_iter):
            # 1) Esplorazione: aggiunta di rumore decrescente nel tempo
            if self.config.token_noise is not None:
                with torch.no_grad():
                    opt_tokens.add_(
                        self.config.token_noise
                        * self._token_noise_schedule(i)
                        * torch.randn_like(opt_tokens)
                    )
            # 2) Forward con autocast opzionale (fp16) per efficienza
            with torch.autocast(
                orig_tokens.device.type,
                torch.float16,
                enabled=self.config.enable_amp,
            ):
                dec = self.decode(opt_tokens)  # decode tokens -> immagine
                loss = self.objective(dec)     # calcolo loss per elemento di batch (b,)
                # 3) Regolarizzazione opzionale per contenere la deriva dei token
                if self.config.reg_weight is not None:
                    assert self.config.reg_type is not None
                    if self.config.reg_type == "seed":
                        reg = self.config.reg_weight * torch.mean(
                            (opt_tokens - orig_tokens) ** 2, dim=(1, 2, 3)
                        )
                    elif self.config.reg_type == "zero":
                        reg = self.config.reg_weight * torch.mean(
                            opt_tokens ** 2, dim=(1, 2, 3)
                        )
                    else:
                        assert False
                else:
                    reg = 0
                # Sommiamo su batch per il passaggio a scaler/optimizer
                sum_loss = torch.sum(loss + reg, dim=0)
            # 4) Backprop con scaler (per AMP) e step optimizer
            # 4) Backprop con scaler (per AMP) e step optimizer
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # 5) Token reset opzionale (es. clamp, proiezione fattoriale, ecc.)
            with torch.no_grad():
                if token_reset_callback is not None and (
                    tokens_reset := token_reset_callback(TestTimeOptInfo(
                        i=i,
                        tokens=opt_tokens,
                        img=dec,
                        loss=loss,
                    ))
                ) is not None:
                    opt_tokens.copy_(tokens_reset.detach())

            # 6) Aggiorna EMA e invoca callback (con media EMA applicata ai token)
            ema.step()
            with ema.average_parameters(), torch.no_grad():
                if callback is not None and callback(TestTimeOptInfo(
                    i=i,
                    tokens=opt_tokens,  # tokens con EMA
                    img=dec,            # immagine decodificata senza EMA sul decoder
                    loss=loss,
                )):
                    break
        # Immagine finale: decode con tokens EMA, clamp in [0,1]
        with ema.average_parameters(), torch.no_grad():
            return torch.clamp(
                self.decode(opt_tokens), 0.0, 1.0
            )


class AugmentationHelper:
    """Wrapper minimal per generare N augmentations coerenti con l'input del modello percettivo.

    - Se `num_augmentations` > 0 applica RandomCrop + Flip; altrimenti restituisce un batch di 1.
    - Ritorno: tensore shape (num_aug, b, c, h, w).
    """
    def __init__(
        self,
        num_augmentations: int,
        img_size,
    ):
        self.num_augmentations = num_augmentations
        if num_augmentations >= 1:
            self.augmentations = v2.Compose([
                v2.RandomCrop(size=img_size),
                v2.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.augmentations = None

    def __call__(
        self,
        x: Float[Tensor, "b c h_in w_in"]
    ) -> Float[Tensor, "num_aug b c h w"]:
        if self.augmentations is None:
            # restituisce una dimensione di "num_aug" = 1
            return x.unsqueeze(0)
        else:
            # genera N viste augmentate della stessa immagine di input
            return torch.stack(
                [self.augmentations(x) for _ in range(self.num_augmentations)]
            )


class CLIPObjective(nn.Module):
    """Obiettivo: massimizzare similarità CLIP tra immagine e prompt testuale.

    - Supporta prompt negativo (neg_prompt) con peso controllato da cfg_scale:
      loss = -(sim_pos) - (1-cfg_scale) * sim_neg
    - Usa augmentations per robustezza (media delle similarità su viste diverse).

    Nota: la loss restituita è per elemento di batch, segno negativo per minimizzazione.
    """
    device_indicator: Tensor

    def __init__(
        self,
        prompt: str | list[str] | None = None,
        neg_prompt: str | list[str] | None = None,
        cfg_scale: float = 1.,
        num_augmentations: int = 0,
        pretrained: tuple[str, str] = ("ViT-B-32", "laion2b_s34b_b79k"),
    ):
        super().__init__()

        # Buffer per derivare il device corrente senza assumere cuda esplicitamente
        self.register_buffer("device_indicator", torch.tensor(0))
        # Carica modello CLIP da open_clip e relative trasformazioni
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            pretrained[0], pretrained=pretrained[1]
        )
        self.clip_tokenizer = cast(open_clip.SimpleTokenizer, open_clip.get_tokenizer("ViT-B-32"))
        self.augment = AugmentationHelper(
            num_augmentations=num_augmentations,
            img_size=self.clip_model.visual.image_size,
        )
        self.eval()

        self._prompt = prompt
        self._prompt_feat = None
        self._neg_prompt = neg_prompt
        self._neg_prompt_feat = None
        # cfg_scale>1 enfatizza il prompt positivo; qui traduciamo in peso del negativo
        self.neg_prompt_weight = 1 - cfg_scale

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        # invalidiamo cache dei feat quando cambia il testo
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        """Ritorna features testuali normalizzate per il prompt corrente (con caching)."""
        assert not self.training
        if self._prompt_feat is None:
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.prompt))
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._prompt_feat = prompt_feat
        return self._prompt_feat

    @property
    def neg_prompt(self):
        return self._neg_prompt

    @neg_prompt.setter
    def neg_prompt(self, prompt):
        self._neg_prompt_feat = None
        self._neg_prompt = prompt

    @property
    @torch.no_grad
    def neg_prompt_feat(self) -> Float[Tensor, "#b d"]:
        """Features testuali normalizzate per il prompt negativo (con caching)."""
        assert not self.training
        if self._neg_prompt_feat is None:
            prompt_feat = self.clip_model.encode_text(self.tokenize(self.neg_prompt))
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)
            self._neg_prompt_feat = prompt_feat
        return self._neg_prompt_feat

    def preprocess(self, img):
        """Applica trasformazioni differenziabili: resize a input CLIP + normalizzazione.

        Nota: estraiamo `resize` e `normalize` dalle transforms di open_clip,
        preservando il resto della pipeline in modo differenziabile.
        """
        resize = self.clip_preprocess.transforms[0]  # type: ignore
        normalize = self.clip_preprocess.transforms[4]  # type: ignore
        if not (img.shape[-1] == img.shape[-2] == resize.size):
            img = F.interpolate(img, size=resize.size, mode="bilinear")
        img = normalize(img)
        return img

    def tokenize(self, text):
        """Tokenizza stringhe/lista di stringhe in input per il CLIP tokenizer selezionato."""
        if isinstance(text, str):
            text = [text]
        return self.clip_tokenizer(text).to(self.device_indicator.device)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        """Calcola loss = -similarità media tra immagine (con augmentations) e prompt.

        - Accumula su `num_augmentations` viste per robustezza; normalizza i vettori immagine/testo.
        - Se definito un `neg_prompt`, sottrae anche il termine negativo pesato.
        """
        assert not self.training
        augs = self.augment(img)  # shape: num_aug x b x c x h x w
        num_augs = augs.shape[0]
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        image_feats = self.clip_model.encode_image(self.preprocess(augs))
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        # similarità media tra features immagine e features testuali (una per elemento del batch)
        similarity = torch.mean(
            einsum(image_feats, self.prompt_feat.mT, "n b d, d b -> n b"),
            dim=0
        )
        if self.neg_prompt is not None:
            neg_similarity = torch.mean(
                einsum(image_feats, self.neg_prompt_feat.mT, "n b d, d b -> n b"),
                dim=0
            )
            return -similarity - self.neg_prompt_weight * neg_similarity
        else:
            return -similarity


class SigLIPObjective(nn.Module):
    """Obiettivo basato su SigLIP (alternativa a CLIP) per similarità immagine-testo.

    Fornisce una loss negativa della similarità calcolata da SigLIP; supporta augmentations.
    """
    def __init__(
        self,
        prompt: str | list[str] | None = None,
        num_augmentations: int = 0,
    ):
        super().__init__()
        self.siglip = SigLIP()
        self.augment = AugmentationHelper(
            num_augmentations=num_augmentations,
            img_size=224,
        )
        self.eval()

        self._prompt = prompt
        self._prompt_feat = None

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, prompt):
        self._prompt_feat = None
        self._prompt = prompt

    @property
    @torch.no_grad
    def prompt_feat(self) -> Float[Tensor, "#b d"]:
        assert not self.training
        if self._prompt_feat is None:
            self._prompt_feat = self.siglip.encode_text(self._prompt)
        return self._prompt_feat

    def preprocess(self, img):
        # Ridimensioniamo a 224 come richiesto da SigLIP
        return F.interpolate(img, size=224, mode="bilinear")

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        assert not self.training
        augs = self.augment(img)  # num_aug b c h w
        num_augs = augs.shape[0]
        augs = rearrange(augs, "n b c h w -> (n b) c h w")
        image_feats = self.siglip.encode_img(self.preprocess(augs), differentiable=True)
        image_feats = rearrange(image_feats, "(n b) d -> n b d", n=num_augs)
        return -torch.mean(
            self.siglip.similarity(
                image_embeds=image_feats,
                text_embeds=self.prompt_feat.unsqueeze(0)
            ),
            dim=0
        )


class MultiObjective(nn.Module):
    """Combinazione lineare di più obiettivi percettivi.

    Esempio: sommare CLIP e SigLIP con pesi diversi per bilanciare preferenze.
    """
    def __init__(self, objectives: list[nn.Module], weights: list[float]):
        super().__init__()
        self.weights = weights
        self.objectives = nn.ModuleList(objectives)

    def forward(self, img: Float[Tensor, "b c h w"]) -> Float[Tensor, "b"]:
        loss = torch.zeros_like(img[:, 0, 0, 0])
        for w, o in zip(self.weights, self.objectives):
            loss = loss + w * o(img)
        return loss
