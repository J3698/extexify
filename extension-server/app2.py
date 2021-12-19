"""
from flask import Flask
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/classify")
@cross_origin()
def classify_symbol():
    return {"top5": ["A", "B", "C", "D", "E"]}
"""


from flask import Flask, request
import json
import sys
from PIL import Image, ImageDraw
import numpy as np
import torch
sys.path.append("..")
#from train2 import Model

#model = Model()
state_dict = torch.load("../Test.pt", map_location = torch.device("cpu"))
print(state_dict['model'])
model.load_state_dict(state_dict)

app = Flask(__name__)

@app.route("/classify")
def classify_symbol():
    points = json.loads(request.args.get('points'))['data']

    if len(points) == 1 and len(points[0]) == 0:
        return {"top3": [" ", " ", " "]}

    image = points_to_image(points)
    print(model(image).shape)

    return {"top3": [" ", " ", " ", " ", " "]}


def points_to_image(points):
    scale_points(points)

    image = Image.new("RGB", (64, 64), color = 0)
    draw = ImageDraw.Draw(image)
    for stroke in points[:-1]:
        for i in range(len(stroke) - 1):
            p1 = 64 * stroke[i][0], 64 * stroke[i][1]
            p2 = 64 * stroke[i + 1][0], 64 * stroke[i + 1][1]
            draw.line(p1 + p2, fill=255, width=1)
        draw.point(stroke[len(stroke) - 1], fill=255)
    image.save("test.png")

    return np.asarray(image).transpose(2, 0, 1)


def scale_points(points):
    min_x = min([point[0] for stroke in points for point in stroke])
    min_y = min([point[1] for stroke in points for point in stroke])
    for stroke in points:
        for point in stroke:
            point[0] -= min_x
            point[1] -= min_y

    max_x = max([point[0] for stroke in points for point in stroke])
    max_y = max([point[1] for stroke in points for point in stroke])
    for stroke in points:
        for point in stroke:
            point[0] /= max_x
            point[1] /= max_y

    for stroke in points:
        for point in stroke:
            point[0] *= 0.95
            point[1] *= 0.95
            point[0] += 0.025
            point[1] += 0.025


