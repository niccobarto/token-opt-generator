from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List

class PanelSaver:
    def __init__(self, caption_height: int = 60, margin: int = 8):
        self.caption_height = caption_height
        self.margin = margin
        # Font di default PIL (niente dipendenze esterne)
        self.font = ImageFont.load_default()

    def _draw_caption(self, canvas: Image.Image, prompt: str) -> None:
        draw = ImageDraw.Draw(canvas)
        # Wrapping semplice
        max_width = canvas.width - 2*self.margin
        lines = self._wrap_text(prompt, draw, max_width)
        y = canvas.height - self.caption_height + self.margin
        for line in lines:
            draw.text((self.margin, y), line, fill=(0,0,0), font=self.font)
            y += self.font.size + 2

    def _wrap_text(self, text: str, draw: ImageDraw.ImageDraw, max_width: int) -> List[str]:
        words = text.split()
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if draw.textlength(test, font=self.font) <= max_width:
                cur = test
            else:
                if cur: lines.append(cur)
                cur = w
        if cur: lines.append(cur)
        return lines

    def make_panel(
        self,
        img_orig: Image.Image,
        img_edit: Image.Image,
        prompt: str,
        out_path: Path,
        max_side: int = 768
    ) -> Path:
        img_o = img_orig.convert("RGB")
        img_e = img_edit.convert("RGB")

        # Ridimensionamento proporzionale
        def resize_max(im: Image.Image, side: int) -> Image.Image:
            w, h = im.size
            scale = side / max(w, h)
            if scale < 1.0:
                im = im.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
            return im

        img_o = resize_max(img_o, max_side)
        img_e = resize_max(img_e, max_side)

        # Uniformiamo l'altezza
        h = max(img_o.height, img_e.height)
        # allineamento verticale centrato
        def pad_to_h(im: Image.Image, H: int) -> Image.Image:
            if im.height == H:
                return im
            top = (H - im.height)//2
            canvas = Image.new("RGB", (im.width, H), (255,255,255))
            canvas.paste(im, (0, top))
            return canvas

        img_o = pad_to_h(img_o, h)
        img_e = pad_to_h(img_e, h)

        # Canvas finale
        W = img_o.width + img_e.width + 3*self.margin
        H = h + self.caption_height + 2*self.margin
        canvas = Image.new("RGB", (W, H), (255,255,255))

        # Paste: [orig]  margin  [edit]
        x = self.margin
        canvas.paste(img_o, (x, self.margin))
        x += img_o.width + self.margin
        canvas.paste(img_e, (x, self.margin))

        # Captions
        self._draw_caption(canvas, prompt)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return out_path