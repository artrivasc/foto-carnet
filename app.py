import os
import base64
import json
import anthropic
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
import io
import concurrent.futures
from rembg import remove, new_session

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MAX_FOTOS = 15
OUTPUT_W = 600
OUTPUT_H = 800

# Load rembg model once at startup (downloads ~170MB on first run)
print("Loading background removal model...")
rembg_session = new_session("u2net_human_seg")
print("Model ready.")


def analizar_con_claude(img_bytes, media_type):
    b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64}
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze this photo to find the person's face for a passport/ID crop.\n"
                        "Return ONLY valid JSON, no markdown:\n"
                        '{"faceFound":true,"cropBox":{"xPct":number,"yPct":number,"wPct":number,"hPct":number},"nota":"frase corta en español"}\n\n'
                        "CRITICAL rules for cropBox:\n"
                        "- xPct/yPct = top-left corner as % of image dimensions (0-100)\n"
                        "- wPct/hPct = crop width/height as % of image dimensions\n"
                        "- Include the FULL head (never cut the hair) + shoulders + upper chest\n"
                        "- Leave at least 10% breathing room above the top of the head\n"
                        "- Leave at least 10% padding on left and right sides\n"
                        "- Eyes should land in the upper 40% of the crop area\n"
                        "- Crop ratio should be close to 3:4 (portrait, taller than wide)\n"
                        "- If face is already very close/zoomed in, use most of the image\n"
                        'If no face: {"faceFound":false,"cropBox":{"xPct":0,"yPct":0,"wPct":100,"hPct":100},"nota":"No se detectó un rostro"}'
                    )
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


def procesar_una_foto(img_bytes, media_type, bg_color):
    # Step 1: Claude finds the face crop
    analysis = analizar_con_claude(img_bytes, media_type)

    if not analysis.get("faceFound"):
        return {
            "ok": False,
            "error": "No se detectó un rostro en esta foto.",
            "nota": analysis.get("nota", "")
        }

    # Step 2: Crop to head + shoulders
    crop = analysis["cropBox"]
    img_original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img_original.size

    x = max(0, int((crop["xPct"] / 100) * w))
    y = max(0, int((crop["yPct"] / 100) * h))
    cw = min(int((crop["wPct"] / 100) * w), w - x)
    ch = min(int((crop["hPct"] / 100) * h), h - y)
    if cw < 10 or ch < 10:
        x, y, cw, ch = 0, 0, w, h

    cropped = img_original.crop((x, y, x + cw, y + ch))

    # Step 3: Remove background with rembg (u2net_human_seg = optimized for people)
    buf_in = io.BytesIO()
    cropped.save(buf_in, format="PNG")
    buf_in.seek(0)

    nobg_bytes = remove(buf_in.read(), session=rembg_session)
    person_rgba = Image.open(io.BytesIO(nobg_bytes)).convert("RGBA")

    # Step 4: Sharpen the person
    r_ch, g_ch, b_ch, a_ch = person_rgba.split()
    rgb = Image.merge("RGB", (r_ch, g_ch, b_ch))
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.0, percent=110, threshold=3))
    rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
    person_sharp = Image.merge("RGBA", (*rgb.split(), a_ch))

    # Step 5: Scale to fit OUTPUT canvas preserving ratio
    pw, ph = person_sharp.size
    scale = min(OUTPUT_W / pw, OUTPUT_H / ph)
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    person_resized = person_sharp.resize((new_w, new_h), Image.LANCZOS)

    # Step 6: Paste centered on solid color background
    bg_r, bg_g, bg_b = hex_to_rgb(bg_color)
    canvas = Image.new("RGBA", (OUTPUT_W, OUTPUT_H), (bg_r, bg_g, bg_b, 255))
    paste_x = (OUTPUT_W - new_w) // 2
    paste_y = (OUTPUT_H - new_h) // 2
    canvas.paste(person_resized, (paste_x, paste_y), mask=person_resized)

    # Convert to RGB JPEG
    result = Image.new("RGB", (OUTPUT_W, OUTPUT_H), (bg_r, bg_g, bg_b))
    result.paste(canvas.convert("RGB"), (0, 0), mask=canvas.split()[3])

    buf_out = io.BytesIO()
    result.save(buf_out, format="JPEG", quality=95)
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(procesar_tarea, t): t[0] for t in tareas}
        for future in concurrent.futures.as_completed(futures):
            nombre, res = future.result()
            resultados[nombre] = res

    return jsonify({"resultados": resultados})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
