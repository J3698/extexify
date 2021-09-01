import torch
from onnx_tf.backend import prepare
from models import Model
import onnx
import onnxruntime
import numpy as np
from tkinter import *
from prep_data import dataloaders
from train import topk_correct
from torch.nn.utils.rnn import pad_packed_sequence


checkpoint = torch.load("./128-2.pt", map_location = torch.device('cpu'))
model = Model(128).float()
model.load_state_dict(checkpoint['model'])
model.eval()

dynamic_axes = { 'input' : {1 : 'seq_length'},\
                'output' : {1 : 'seq_length'}}

torch.onnx.export(model,
                  torch.ones((10, 1, 3,)),      # model input
                  "128-torch_model2.onnx",
                  export_params = True,
                  opset_version = 10,
                  do_constant_folding = True,   # for optimization
                  input_names = ['input'],      # model's input names
                  dynamic_axes = dynamic_axes)

onnx_model = onnx.load("128-torch_model2.onnx")
onnx.checker.check_model(onnx_model)
tf_rep = prepare(onnx_model)
tf_rep.export_graph("128-torch_model2.pb")
"""
ort_session = onnxruntime.InferenceSession("128-torch_model2.onnx")

inp = torch.randn((1, 10, 3,))
out = ort_session.run(None, {ort_session.get_inputs()[0].name: inp.numpy()})
out2 = model(inp).detach().numpy()
"""

