
from darknet import Darknet

import torch.onnx
from torch.autograd import Variable



import cv2
import numpy as np
from numpy import array

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common


cfgfile =  './multi_obj_pose_estimation/cfg/yolo-pose-multi.cfg' 
weightfile =  './multi_obj_pose_estimation/backup_multi/c920_cam/brownGlyphClassSetToZero/model.weights'

m = Darknet(cfgfile)
m.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))

# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = Variable(torch.randn(1, 3, 416, 416))
# Obtain your model, it can be also constructed in your script explicitly
model = m
# Invoke export
torch.onnx.export(model, dummy_input, './trt_models/multi_objs/multiobj_cargo_59_hatchPanel_75.onnx')



# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Try to load a previously generated yolo network graph in ONNX format:
onnx_file_path = './trt_models/multi_objs/multiobj_cargo_59_hatchPanel_75.onnx'
engine_file_path = './trt_models/multi_objs/multiobj_cargo_59_hatchPanel_75.trt'
input_image_path = './hatchPanel_sample.jpg'

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                #parser.parse(model.read())
                #parser.parse returns a bool, and we were not checking it originally.
                if not parser.parse(model.read()):
                    print(parser.get_error(0))
                print(network.get_layer(network.num_layers - 1).get_output(0).shape)
            network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))  

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            print(engine)
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


# Do inference with TensorRT
trt_outputs = []
with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    print(type(engine))
    img = preprosess_img(input_image_path)
    # Do inference
    print('Running inference on image {}...'.format(input_image_path))
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    inputs[0].host = img
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(trt_outputs)
    print(type(trt_outputs))