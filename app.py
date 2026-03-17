import os
import base64
import json
import anthropic
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import io
import concurrent.futures

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MAX_FOTOS = 15
OUTPUT_W = 600
OUTPUT_H = 800


def analizar_con_claude(img_bytes, media_type):
    """Single Claude call: get crop box + person outline polygon."""
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64}
                },
                {
                    "type": "text",
                    "text": """Analyze this portrait photo for passport/ID processing.
Return ONLY valid JSON, no markdown, no explanation:

{
  "faceFound": true,
  "cropBox": {
    "xPct": number,
    "yPct": number,
    "wPct": number,
    "hPct": number
  },
  "personOutline": [[x1Pct,y1Pct],[x2Pct,y2Pct],...],
  "nota": "frase corta en español"
}

Rules for cropBox (percentages of full image size 0-100):
- Include full head + shoulders + upper chest
- 10% padding above head, 10% on each side
- Eyes in upper 40% of crop area
- Portrait ratio ~3:4

Rules for personOutline:
- A polygon that traces the OUTLINE of the person (head + shoulders + body visible)
- 20 to 40 points as [xPct, yPct] percentages of the FULL image dimensions
- Trace carefully around hair, ears, neck, shoulders
- The polygon should separate person from background as accurately as possible
- Go clockwise starting from top of head

If no face found:
{"faceFound":false,"cropBox":{"xPct":0,"yPct":0,"wPct":100,"hPct":100},"personOutline":[],"nota":"No se detectó un rostro"}"""
                }
            ]
        }]
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def hex_to_rgb(hex_color):
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def apply_person_mask(img, outline_pcts, bg_color_rgb):
    """
    Use Claude's polygon outline to mask the background.
    Returns RGB image with background replaced by bg_color.
    """
    w, h = img.size

    if not outline_pcts or len(outline_pcts) < 3:
        # No valid outline — return image as-is
        return img

    # Convert percentage points to pixel coordinates
    poly = [(int(p[0] / 100 * w), int(p[1] / 100 * h)) for p in outline_pcts]

    # Create mask: white where person is, black where background is
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(poly, fill=255)

    # Feather the mask edges slightly for smoother look
    from PIL import ImageFilter as IF
    mask = mask.filter(IF.GaussianBlur(radius=3))

    # Composite: person over solid background
    bg = Image.new("RGB", (w, h), bg_color_rgb)
    result = Image.composite(img, bg, mask)
    return result


def procesar_una_foto(img_bytes, media_type, bg_color):
    # Step 1: Claude analyzes — crop + outline in one call
    analysis = analizar_con_claude(img_bytes, media_type)

    if not analysis.get("faceFound"):
        return {
            "ok": False,
            "error": "No se detectó un rostro en esta foto.",
            "nota": analysis.get("nota", "")
        }

    bg_rgb = hex_to_rgb(bg_color)

    # Step 2: Open image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    # Step 3: Apply person mask to full image first
    outline = analysis.get("personOutline", [])
    img_masked = apply_person_mask(img, outline, bg_rgb)

    # Step 4: Crop to head+shoulders area
    crop = analysis["cropBox"]
    x = max(0, int((crop["xPct"] / 100) * w))
    y = max(0, int((crop["yPct"] / 100) * h))
    cw = min(int((crop["wPct"] / 100) * w), w - x)
    ch = min(int((crop["hPct"] / 100) * h), h - y)
    if cw < 20 or ch < 20:
        x, y, cw, ch = 0, 0, w, h

    cropped = img_masked.crop((x, y, x + cw, y + ch))

    # Step 5: Enhance
    cropped = cropped.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
    cropped = ImageEnhance.Contrast(cropped).enhance(1.05)
    cropped = ImageEnhance.Brightness(cropped).enhance(1.02)

    # Step 6: Scale to output canvas
    cw2, ch2 = cropped.size
    scale = min(OUTPUT_W / cw2, OUTPUT_H / ch2)
    new_w = int(cw2 * scale)
    new_h = int(ch2 * scale)
    cropped = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Step 7: Center on solid background
    canvas = Image.new("RGB", (OUTPUT_W, OUTPUT_H), bg_rgb)
    paste_x = (OUTPUT_W - new_w) // 2
    paste_y = (OUTPUT_H - new_h) // 2
    canvas.paste(cropped, (paste_x, paste_y))

    buf_out = io.BytesIO()
    canvas.save(buf_out, format="JPEG", quality=95)
    buf_out.seek(0)
    resultado_b64 = base64.standard_b64encode(buf_out.read()).decode("utf-8")

    return {
        "ok": True,
        "imagen": resultado_b64,
        "nota": analysis.get("nota", "Procesada correctamente")
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/procesar", methods=["POST"])
def procesar():
    fotos = request.files.getlist("fotos")
    if not fotos:
        return jsonify({"error": "No se recibieron fotos"}), 400
    if len(fotos) > MAX_FOTOS:
        return jsonify({"error": f"Máximo {MAX_FOTOS} fotos por vez"}), 400

    bg_color = request.form.get("fondo", "#FFFFFF")

    tareas = []
    for foto in fotos:
        img_bytes = foto.read()
        media_type = foto.content_type or "image/jpeg"
        if media_type not in ["image/jpeg", "image/png", "image/webp"]:
            media_type = "image/jpeg"
        tareas.append((foto.filename, img_bytes, media_type))

    resultados = {}

    def procesar_tarea(tarea):
        nombre, img_bytes, media_type = tarea
        try:
            res = procesar_una_foto(img_bytes, media_type, bg_color)
            res["nombre"] = nombre
            return nombre, res
        except Exception as e:
            return nombre, {"ok": False, "error": str(e), "nombre": nombre}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(procesar_tarea, t): t[0] for t in tareas}
        for future in concurrent.futures.as_completed(futures):
            nombre, res = future.result()
            resultados[nombre] = res

    return jsonify({"resultados": resultados})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
