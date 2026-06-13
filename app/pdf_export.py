from __future__ import annotations

from pathlib import Path
import re

from PIL import Image, ImageOps


def normalize_pdf_filename(filename: str) -> str:
    name = Path(str(filename).strip()).name
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .")
    if not name:
        name = "mosaic_images.pdf"
    if Path(name).suffix.lower() != ".pdf":
        name += ".pdf"
    return name


def create_image_pdf(image_paths: list[str | Path], destination: str | Path) -> tuple[Path, int]:
    paths = [Path(path) for path in image_paths if Path(path).is_file()]
    if not paths:
        raise ValueError("PDFにまとめる画像がありません。")

    pages: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source)
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    page = Image.new("RGB", rgba.size, "white")
                    page.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    page = image.convert("RGB")
                pages.append(page.copy())

        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp.pdf")
        pages[0].save(temp, "PDF", save_all=True, append_images=pages[1:], resolution=100.0)
        temp.replace(output)
        return output, len(pages)
    finally:
        for page in pages:
            page.close()
