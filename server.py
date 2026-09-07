import os
import threading
import uuid

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect
)

from generate import generate


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

TRAINING_DIR = os.path.join(
    BASE_DIR,
    "training_data"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    TRAINING_DIR,
    exist_ok=True
)


app = Flask(__name__)


generation_lock = threading.Lock()


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route(
    "/generate",
    methods=["POST"]
)
def generate_video():

    prompt = request.form.get(
        "prompt",
        ""
    ).strip()

    if not prompt:

        return redirect("/")

    # Не позволяем одновременно
    # запускать несколько тяжёлых
    # генераций на CPU.

    if not generation_lock.acquire(
        blocking=False
    ):

        return (
            "TinyVideo уже генерирует "
            "другое видео. Подожди.",
            429
        )

    try:

        generate(prompt)

    finally:

        generation_lock.release()

    return redirect("/")


@app.route(
    "/feedback",
    methods=["POST"]
)
def feedback():

    prompt = request.form.get(
        "prompt",
        ""
    ).strip()

    rating = request.form.get(
        "rating",
        ""
    )

    if prompt and rating:

        filename = os.path.join(
            TRAINING_DIR,
            "feedback.txt"
        )

        with open(
            filename,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"{rating}\t{prompt}\n"
            )

    return redirect("/")


@app.route("/video")
def video():

    return send_from_directory(
        OUTPUT_DIR,
        "generated.mp4"
    )


if __name__ == "__main__":

    print()
    print("=" * 55)
    print("                  TinyVideo Server")
    print("=" * 55)
    print()
    print("Local server:")
    print("http://127.0.0.1:5000")
    print()
    print("Press CTRL+C to stop.")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
