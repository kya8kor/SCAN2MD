"""
Scan2MD — Scanned PDF to Markdown converter.

Pipeline:
  1. Rasterize each PDF page to a high-res image (PyMuPDF).
  2. Send the page image to Gemini vision with a prompt asking for a
     structured JSON list of content blocks (heading/paragraph/table/figure)
     IN READING ORDER, each with a bounding box.
  3. Crop real pixels for every "figure" block straight from the rasterized
     page (never let the model regenerate images) and base64-encode them.
  4. Walk the blocks in order and emit Markdown, with figures embedded
     inline as base64 data-URIs.

Markdown can't preserve absolute x/y position — it's a linear format. What
this preserves is reading order, structural hierarchy, and correctly-placed
embedded images, which is the closest a Markdown file can get to "looking
like" the original scan.
"""

import base64
import io
import json
import os
import re

import pypdfium2 as pdfium
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from google import genai
from google.genai import types

APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder="templates", static_url_path="")

GEMINI_MODEL = "gemini-3.1-flash-lite"
RENDER_DPI = 300

BLOCK_PROMPT = """You are analyzing one page of a SCANNED document image (no
selectable text layer exists, this is a flat picture of a page).

Return ONLY valid JSON (no markdown fences, no commentary) with this shape:

{
  "blocks": [
    {"type": "heading", "level": 1, "text": "...", "box_2d": [ymin,xmin,ymax,xmax]},
    {"type": "paragraph", "text": "...", "box_2d": [ymin,xmin,ymax,xmax]},
    {"type": "list_item", "text": "...", "box_2d": [ymin,xmin,ymax,xmax]},
    {"type": "table", "markdown": "| a | b |\\n|---|---|\\n| 1 | 2 |", "box_2d": [ymin,xmin,ymax,xmax]},
    {"type": "figure", "caption": "verbatim printed caption/label text, or empty string if none", "box_2d": [ymin,xmin,ymax,xmax]}
  ]
}

Rules:
- "blocks" MUST be ordered exactly as a human would read the page
  (top-to-bottom, left-to-right; respect multi-column layouts by finishing
  one column before the next).
- box_2d is [ymin, xmin, ymax, xmax] on a NORMALIZED 0-1000 scale, where
  (0,0) is the top-left corner of the image and (1000,1000) is the
  bottom-right corner. Use exactly this field name and exactly this
  y-before-x axis order (the standard object-detection convention) —
  do not reorder it to x-first and do not rename the field to "bbox".
- Use "figure" for any photo, chart, diagram, logo, or illustration.
  Do not use "figure" for plain text, even if stylized. Make the figure
  box_2d tight around just the visual (do not include surrounding text or
  its caption). This includes pie charts, bar charts, and graphs — even
  when they sit in their own narrow column beside body text. Every visual
  element on the page must get its own figure block; do not skip one
  because the page is dense or the layout is multi-column.
- If the figure has a printed caption or label near it (e.g. "Figure 1:
  ..." or a title under a chart), copy that caption text VERBATIM into
  "caption" — do not paraphrase it or invent your own description. If the
  figure has no printed caption anywhere near it, use an empty string.
- Do NOT merge multiple visually distinct paragraphs, headings, or list
  items into a single block just because they sit close together with no
  large vertical gap. Each numbered or lettered marker (e.g. "1)", "2)",
  "a)", bullet symbols) starts a NEW block — either "heading" or
  "list_item" — separate from the paragraph(s) that follow it, even when
  there is no visible blank line before it. A page with two numbered
  sections and two paragraphs must produce at least four text blocks, not
  one block containing all of it.
- Reproduce all text VERBATIM as best you can read it (this is OCR).
  Preserve the exact bracket characters used (e.g. plain [ ] must stay as
  [ ], do not convert to full-width brackets), and preserve line breaks
  within a single multi-line title as one heading block.
- Use "table" only for genuine tabular/grid data; give it as a markdown
  table string.
- heading level is 1-6 based on visual prominence (font size/boldness).
- If the page is blank or has no content, return {"blocks": []}.
"""


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export your Gemini API key before starting the server."
        )
    return genai.Client(api_key=api_key)


def render_page(doc, page_index, dpi=RENDER_DPI):
    page = doc[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    page.close()
    return pil_image.convert("RGB")


def extract_json(raw_text):
    """Gemini sometimes wraps JSON in fences despite instructions; strip them.
    It also occasionally tacks on trailing whitespace/text after a otherwise
    complete JSON object (json.loads rejects that outright with "Extra data"
    instead of ignoring it), so parse with raw_decode and only take the
    first valid JSON value, discarding anything after it."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    obj, _end_index = decoder.raw_decode(cleaned)
    return obj


def analyze_page(client, page_image):
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[BLOCK_PROMPT, page_image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = extract_json(response.text)
    blocks = data.get("blocks", [])
    for block in blocks:
        block["bbox"] = normalize_box(block)

    # DEBUG: print a one-line summary per block so we can see exactly what
    # Gemini classified this page as, without needing to inspect the final
    # rendered markdown. Remove once figure/segmentation issues are sorted.
    print(f"--- analyze_page: {len(blocks)} block(s) ---")
    for i, b in enumerate(blocks):
        btype = b.get("type")
        bbox = b.get("bbox")
        if btype == "figure":
            preview = f'caption={b.get("caption")!r} bbox={bbox}'
        else:
            text = (b.get("text") or b.get("markdown") or "")[:60]
            preview = f'text={text!r} bbox={bbox}'
        print(f"  [{i}] {btype}: {preview}")

    return blocks


def normalize_box(block):
    """Gemini's vision models are natively trained on [ymin,xmin,ymax,xmax]
    (y-before-x, the standard object-detection convention) regardless of
    what field name/order you ask for — asking it to output "bbox":
    [x0,y0,x1,y1] gets silently ignored in practice and it keeps returning
    y-first values under whatever key you gave it, which produces crops
    with width/height swapped (tall skinny strips instead of the real
    region). The prompt now explicitly asks for the native "box_2d":
    [ymin,xmin,ymax,xmax] format and this converts it to the [x0,y0,x1,y1]
    order the rest of the pipeline expects.
    """
    box = block.get("box_2d") or block.get("bbox")
    if not box or len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = box
    return [xmin, ymin, xmax, ymax]


MAX_FIGURE_WIDTH = 700   # px — figures are downscaled to this before embedding
JPEG_QUALITY = 78        # good size/quality tradeoff for embedded base64 images


def crop_figure(page_image, bbox):
    """bbox is on a normalized 0-1000 scale (Gemini's native box format);
    scale it to this page image's actual pixel dimensions before cropping.
    Output is downscaled + JPEG-compressed to keep the embedded base64
    string (and therefore the .md file) small."""
    w, h = page_image.size
    x0, y0, x1, y1 = bbox
    x0 = x0 / 1000.0 * w
    x1 = x1 / 1000.0 * w
    y0 = y0 / 1000.0 * h
    y1 = y1 / 1000.0 * h
    # clamp + sanitize
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None
    crop = page_image.crop((x0, y0, x1, y1))

    # Downscale large figures so the base64 payload stays reasonable.
    if crop.width > MAX_FIGURE_WIDTH:
        scale = MAX_FIGURE_WIDTH / crop.width
        crop = crop.resize(
            (MAX_FIGURE_WIDTH, max(1, int(crop.height * scale))),
            Image.LANCZOS,
        )

    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "jpeg"


def blocks_to_markdown(blocks, page_image, page_number):
    """Renders the page as normal top-to-bottom HTML flow (headings,
    paragraphs, tables, images) EMBEDDED DIRECTLY in the Markdown file.
    Markdown can't hold true x/y position, but Markdown files can contain
    raw HTML, and most renderers (VS Code preview, GitHub, browsers)
    render that HTML.

    Earlier versions of this function pinned every block to Gemini's own
    bounding box (position:absolute). That looked closer to the source
    layout in theory, but Gemini's boxes are frequently oversized for the
    actual text/figure inside them — sometimes dramatically so — which
    produced large dead gaps that no amount of gap-compression could fully
    fix, since the excess space was baked into individual box HEIGHTS, not
    just the gaps between them.

    This version uses normal document flow instead: tight, predictable
    browser-default spacing, guaranteed no oversized gaps. Figures are
    floated left/right based on which side of the page their bbox sits on
    (a reasonably reliable signal even when box height isn't), so images
    still land near where they occurred in reading order and text can
    wrap around them like a real page — just without pixel-exact position.
    """
    parts = [
        f'<div class="s2m-page" data-page="{page_number}" '
        f'style="max-width:900px;margin:0 auto 24px;padding:28px 32px;'
        f'background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.2);'
        f'overflow:auto;font-family:Georgia,\'Times New Roman\',serif;">'
    ]

    for block in blocks:
        btype = block.get("type")

        if btype == "heading":
            level = max(1, min(6, int(block.get("level", 2))))
            size = HEADING_FONT_SIZE.get(level, "1.2em")
            text = _esc(block.get("text", ""))
            parts.append(
                f'<div style="font-size:{size};font-weight:700;color:#111;'
                f'line-height:1.2;margin:0.7em 0 0.35em;clear:both;">{text}</div>'
            )
        elif btype in ("paragraph", "list_item"):
            text = _esc(block.get("text", ""))
            prefix = "• " if btype == "list_item" else ""
            parts.append(
                f'<div style="font-size:0.95em;color:#222;line-height:1.45;'
                f'margin:0 0 0.5em;">{prefix}{text}</div>'
            )
        elif btype == "table":
            table_md = block.get("markdown", "")
            rows = [r for r in table_md.strip().split("\n") if r.strip().startswith("|")]
            rows = [r for r in rows if not re.match(r"^\|[\s:\-|]+\|$", r)]
            table_html = (
                "<table style='border-collapse:collapse;font-size:0.85em;"
                "width:100%;margin:0.5em 0;clear:both;'>"
            )
            for i, row in enumerate(rows):
                cells = [c.strip() for c in row.split("|")[1:-1]]
                tag = "th" if i == 0 else "td"
                table_html += "<tr>" + "".join(
                    f"<{tag} style='border:1px solid #ccc;padding:3px 8px;'>{_esc(c)}</{tag}>"
                    for c in cells
                ) + "</tr>"
            table_html += "</table>"
            parts.append(table_html)
        elif btype == "figure":
            bbox = block.get("bbox")
            b64, fmt = crop_figure(page_image, bbox) if bbox else (None, None)
            if b64:
                x0 = bbox[0] if bbox else 500
                if x0 > 550:
                    float_style = "float:right;margin:4px 0 10px 16px;max-width:46%;"
                elif x0 < 250:
                    float_style = "float:left;margin:4px 16px 10px 0;max-width:46%;"
                else:
                    float_style = "display:block;margin:10px auto;max-width:70%;"
                caption = _esc((block.get("caption") or "").strip())
                caption_html = (
                    f'<figcaption style="font-size:0.82em;color:#555;'
                    f'font-style:italic;line-height:1.3;margin-top:4px;">'
                    f'{caption}</figcaption>'
                    if caption else ""
                )
                # Caption sits inside the same <figure> as the image, so it
                # floats/positions as one unit and always stays attached to
                # its image instead of drifting off as a separate paragraph.
                parts.append(
                    f'<figure style="{float_style}margin:4px 0 10px;">'
                    f'<img src="data:image/{fmt};base64,{b64}" '
                    f'style="display:block;width:100%;height:auto;'
                    f'border-radius:2px;margin:0;" />'
                    f'{caption_html}'
                    f'</figure>'
                )

    parts.append('<div style="clear:both;"></div></div>')
    return "".join(parts)


def _esc(text):
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


HEADING_FONT_SIZE = {1: "2.1em", 2: "1.6em", 3: "1.3em", 4: "1.15em", 5: "1.05em", 6: "1em"}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    upload = request.files["file"]
    if not upload.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400

    try:
        client = get_client()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    pdf_bytes = upload.read()

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
        page_markdowns = []
        for i in range(len(doc)):
            page_image = render_page(doc, i)
            blocks = analyze_page(client, page_image)
            page_md = blocks_to_markdown(blocks, page_image, i + 1)
            page_markdowns.append(f"<!-- Page {i + 1} -->\n\n{page_md}")
        doc.close()

        full_markdown = "\n\n".join(page_markdowns)
        return jsonify({"markdown": full_markdown, "pages": len(page_markdowns)})
    except Exception as exc:  # surfaced to the frontend for visibility
        return jsonify({"error": f"Conversion failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
