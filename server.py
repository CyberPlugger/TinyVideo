import os
import uuid

from flask import Flask, jsonify, request, send_from_directory

from generate import generate, generate_with_image


app = Flask(__name__)

OUTPUT_DIR = "outputs"
TRAINING_DIR = "training_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)


def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type"
    )
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )
    return response


@app.after_request
def after_request(response):
    return add_cors(response)


@app.route("/")
def index():
    return jsonify(
        {
            "name": "TinyVideo",
            "version": "1.4",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "generate": "/generate",
                "generate_with_image": "/generate_with_image",
                "feedback": "/feedback",
            },
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "TinyVideo",
            "version": "1.4",
        }
    )


@app.route("/generate", methods=["POST", "OPTIONS"])
def generate_video():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}

    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify(
            {
                "error": "prompt is required"
            }
        ), 400

    seed = data.get("seed")
    steps = data.get("steps", 20)
    resolution = data.get("resolution")

    try:
        if seed is not None:
            seed = int(seed)

        steps = int(steps)

        if resolution:
            resolution = str(resolution)

        filename = (
            f"video_{uuid.uuid4().hex}.mp4"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        result = generate(
            prompt=prompt,
            seed=seed,
            steps=steps,
            resolution=resolution,
            output=output_path,
        )

        return jsonify(
            {
                "success": True,
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "resolution": resolution,
                "filename": filename,
                "video": f"/video/{filename}",
                "path": result,
            }
        )

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


@app.route(
    "/generate_with_image",
    methods=["POST", "OPTIONS"],
)
def generate_image_video():
    if request.method == "OPTIONS":
        return "", 204

    image = request.files.get("image")

    if image is None:
        return jsonify(
            {
                "error": "image is required"
            }
        ), 400

    prompt = request.form.get(
        "prompt",
        "",
    ).strip()

    if not prompt:
        return jsonify(
            {
                "error": "prompt is required"
            }
        ), 400

    try:
        seed_value = request.form.get("seed")
        steps_value = request.form.get(
            "steps",
            "20",
        )

        resolution = request.form.get(
            "resolution"
        )

        strength_value = request.form.get(
            "strength",
            "0.75",
        )

        seed = (
            int(seed_value)
            if seed_value
            else None
        )

        steps = int(steps_value)
        strength = float(strength_value)

        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                "strength must be between 0 and 1"
            )

        temporary_name = (
            f"input_{uuid.uuid4().hex}"
            + os.path.splitext(
                image.filename or ".png"
            )[1]
        )

        temporary_path = os.path.join(
            TRAINING_DIR,
            temporary_name,
        )

        image.save(temporary_path)

        filename = (
            f"image_video_{uuid.uuid4().hex}.mp4"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        try:
            result = generate_with_image(
                prompt=prompt,
                image_path=temporary_path,
                seed=seed,
                steps=steps,
                resolution=resolution,
                strength=strength,
                output=output_path,
            )
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

        return jsonify(
            {
                "success": True,
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "resolution": resolution,
                "strength": strength,
                "filename": filename,
                "video": f"/video/{filename}",
                "path": result,
            }
        )

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json(silent=True) or {}

    prompt = str(
        data.get("prompt", "")
    ).replace("\n", " ").strip()

    rating = data.get("rating")

    if rating is None:
        return jsonify(
            {
                "error": "rating is required"
            }
        ), 400

    feedback_path = os.path.join(
        TRAINING_DIR,
        "feedback.txt",
    )

    with open(
        feedback_path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"{rating}\t{prompt}\n"
        )

    return jsonify(
        {
            "success": True
        }
    )


@app.route("/video/<path:filename>")
def video(filename):
    return send_from_directory(
        OUTPUT_DIR,
        filename,
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify(
        {
            "error": "not found"
        }
    ), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify(
        {
            "error": "internal server error"
        }
    ), 500


if __name__ == "__main__":
    print()
    print("================================")
    print("TinyVideo V1.4")
    print("================================")
    print("Server: http://127.0.0.1:5000")
    print("API ready.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )