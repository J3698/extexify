from flask import Flask, request
import json
import sys
from PIL import Image, ImageDraw
import numpy as np


app = Flask(__name__)

@app.route("/classify")
def classify_symbol():
    points = json.loads(request.args.get('points'))['data']
    image = points_to_image(points)

    return {"top3": ["A", "B", "C"]}


def points_to_image(points):
    image = Image.new("L", (64, 64), color = 0)
    draw = ImageDraw.Draw(image)
    for stroke in points:
        for i in range(len(stroke) - 1):
            p1 = stroke[i]
            p2 = stroke[i + 1]
            draw.line(p1 + p2, fill=255, width=1)

    return np.array(image.getdata()).reshape((64, 64))
