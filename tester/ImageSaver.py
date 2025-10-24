from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def _wrap_text_by_width(draw, text, font, max_width):
    """Suddivide il testo in righe che stiano dentro max_width (in pixel)."""
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], []
    for w in words:
        trial = " ".join(cur + [w])
        if draw.textlength(trial, font=font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def create_side_by_side_with_caption(
    left_img: Image.Image,
    right_img: Image.Image,
    prompt: str,
    value,
    out_path: str | Path = "output.png",
    target_height: int = 512,
    padding: int = 16,
    gap: int = 16,
    bg_color=(255, 255, 255),
    text_color=(0, 0, 0),
    font_path: str | Path | None = None,
    base_font_size: int = 20,
) -> Image.Image:
    """
    Crea un'immagine con le due immagini affiancate e una didascalia in basso
    contenente `prompt` e `value`. Salva su `out_path` e restituisce un oggetto PIL.Image.Image.
    """
    assert isinstance(left_img, Image.Image), "left_img deve essere un oggetto PIL.Image.Image"
    assert isinstance(right_img, Image.Image), "right_img deve essere un oggetto PIL.Image.Image"

    left = left_img.convert("RGB")
    right = right_img.convert("RGB")

    # --- Ridimensiona mantenendo aspect ratio alla stessa altezza
    def resize_to_height(img, h):
        w, old_h = img.size
        new_w = int(w * (h / old_h))
        return img.resize((new_w, h), Image.Resampling.LANCZOS)

    left_r  = resize_to_height(left, target_height)
    right_r = resize_to_height(right, target_height)

    # --- Dimensioni del blocco immagini
    img_block_width  = left_r.width + gap + right_r.width
    img_block_height = target_height

    # --- Font
    if font_path is not None:
        font = ImageFont.truetype(str(font_path), size=base_font_size)
    else:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=base_font_size)
        except Exception:
            font = ImageFont.load_default()

    # --- Prepara testo didascalia
    footer_text = f"{prompt}  |  CLIPScore = {value}"
    tmp_img = Image.new("RGB", (10, 10))
    tmp_draw = ImageDraw.Draw(tmp_img)
    wrapped_lines = _wrap_text_by_width(tmp_draw, footer_text, font, img_block_width)
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    caption_height = line_height * len(wrapped_lines)

    # --- Canvas finale
    total_width  = img_block_width + 2 * padding
    total_height = padding + img_block_height + gap + caption_height + padding

    canvas = Image.new("RGB", (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    # --- Posiziona le immagini
    x0 = (total_width - img_block_width) // 2
    y0 = padding
    canvas.paste(left_r,  (x0, y0))
    canvas.paste(right_r, (x0 + left_r.width + gap, y0))

    # --- Scrivi la caption centrata sotto
    caption_top = y0 + img_block_height + gap
    cur_y = caption_top
    for line in wrapped_lines:
        line_w = draw.textlength(line, font=font)
        text_x = (total_width - line_w) // 2
        draw.text((text_x, cur_y), line, fill=text_color, font=font)
        cur_y += line_height

    # --- Salva e ritorna
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas
