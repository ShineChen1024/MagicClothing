import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference


class Parsing:
    def __init__(self):
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.lip_session = ort.InferenceSession(os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_lip.onnx'),
                                                sess_options=session_options, providers=['CPUExecutionProvider'])

    def __call__(self, input_image):
        cloth_mask = onnx_inference(self.lip_session, input_image)
        return cloth_mask
