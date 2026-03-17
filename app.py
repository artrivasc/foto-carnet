import os
import base64
import json
import anthropic
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageFilter, ImageEnhance
import io
import concurrent.futures

app = Flask(__name__)
CORS(app)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MAX_FOTOS = 15


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
                        "Analyze this photo for an ID/passport crop. "
                        "Return ONLY valid JSON, no markdown, no explanation:\n"
                        '{"faceFound":true,"cropBox":{"xPct":number,"yPct":number,"wPct":number,"hPct":number},"nota":"frase corta en español"}\n\n'
                        "Rules:\n"
                        "- xPct/yPct = top-left corner as % of image dimensions (0-100)\n"
                        "- wPct/hPct = crop size as % of image dimensions\n"
                        "- Include full head + shoulders (about 25% of torso)\n"
                        "- Add padding: at least 12% on each side of face, 8% above head\n"
                        "- Eyes in upper 45% of crop\n"
                        "- Portrait orientation (height >= width)\n"
                        'If no face: {"faceFound":false,"cropBox":{"xPct":0,"yPct":0,"wPct":100,"hPct":100},"nota":"No se detectó un rostro"}'
                    )
                }
            ]
        }]
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    clean = text.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def procesar_una_foto(img_bytes, media_type, bg_color, output_size):
    analysis = analizar_con_claude(img_bytes, media_type)

    if not analysis.get("faceFound"):
        return {
            "ok": False,
            "error": "No se detectó un rostro en esta foto.",
            "nota": analysis.get("nota", "")
        }

    crop = analysis["cropBox"]
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    x = max(0, int((crop["xPct"] / 100) * w))
    y = max(0, int((crop["yPct"] / 100) * h))
    cw = min(int((crop["wPct"] / 100) * w), w - x)
    ch = min(int((crop["hPct"] / 100) * h), h - y)

    cropped = img.crop((x, y, x + cw, y + ch))
    cropped = cropped.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    cropped = ImageEnhance.Contrast(cropped).enhance(1.08)
    cropped = ImageEnhance.Brightness(cropped).enhance(1.03)

    out_h = int(output_size * (ch / cw))
    cropped = cropped.resize((output_size, out_h), Image.LANCZOS)

    hex_clean = bg_color.lstrip("#")
    r, g, b = int(hex_clean[0:2], 16), int(hex_clean[2:4], 16), int(hex_clean[4:6], 16)
    canvas = Image.new("RGB", (output_size, out_h), (r, g, b))
    canvas.paste(cropped, (0, 0))

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    resultado_b64 = base64.standard_b64encode(buf.read()).decode("utf-8")

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
    output_size = min(int(request.form.get("tamano", "800")), 1200)

    # Read all files into memory before threading
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
            res = procesar_una_foto(img_bytes, media_type, bg_color, output_size)
            res["nombre"] = nombre
            return nombre, res
        except Exception as e:
            return nombre, {"ok": False, "error": str(e), "nombre": nombre}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_FOTOS) as executor:
        futures = {executor.submit(procesar_tarea, t): t[0] for t in tareas}
        for future in concurrent.futures.as_completed(futures):
            nombre, res = future.result()
            resultados[nombre] = res

    return jsonify({"resultados": resultados})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
