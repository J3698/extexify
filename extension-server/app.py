from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import sys
from PIL import Image, ImageDraw
import numpy as np
import torch
sys.path.append("..")
from train2 import Model
from torchvision.transforms import ToTensor

model = Model()
model.eval()
state_dict = torch.load("Test.pt", map_location = torch.device("cpu"))
model.load_state_dict(state_dict['model'])
size = 32

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

order = sorted([str(i) for i in range(1098)])
chars = sorted(set(np.load("data_processed/dataY.npy")))
def fix_predictions(output):
    outs = [chars[int(order[i.item()])] for i in torch.topk(output, 5, dim = 1).indices[0]]
    return ["\\" + i.split("_")[1] for i in outs]

@app.route("/classify")
@cross_origin()
def classify_symbol():
    json_str = request.args.get('points')
    if json_str is None:
        return {"top5": [" ", " ", " ", " ", " "]}

    json_data = json.loads(json_str)
    if 'data' not in json_data:
        return {"top5": [" ", " ", " ", " ", " "]}

    points = json_data['data']

    if len(points) == 1 and len(points[0]) == 0:
        return {"top5": [" ", " ", " ", " ", " "]}

    points_to_image(points)
    tensor = ToTensor()(Image.open("test.png").convert('RGB'))
    tensor = torch.unsqueeze(tensor, 0)
    output = model(tensor)
    pred = fix_predictions(output)

    return {"top5": pred}


def points_to_image(points):
    scale_points(points)

    image = Image.new("RGB", (size, size), color = 0)
    draw = ImageDraw.Draw(image)
    for stroke in points[:-1]:
        for i in range(len(stroke) - 1):
            p1 = size * stroke[i][0], size * stroke[i][1]
            p2 = size * stroke[i + 1][0], size * stroke[i + 1][1]
            draw.line(p1 + p2, fill=(255, 255, 255), width=1)
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


