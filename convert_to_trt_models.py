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
import matplotlib
import matplotlib.pyplot as plt


cfgfile =  './multi_obj_pose_estimation/cfg/yolo-pose-multi.cfg' 
weightfile =  './multi_obj_pose_estimation/backup_multi/model.weights'

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
torch.onnx.export(model, dummy_input, './trt_models/multi_objs/FRC2020models_v11_powercell_powerport.onnx')




# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        #EXPLICIT_BATCH = 1 << (int)(trt.BuilderFlag.FP16) | 1 << (int)(trt.BuilderFlag.STRICT_TYPES)
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        #with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            if (builder.platform_has_fast_fp16):
                print('support fp16')
            if (builder.platform_has_fast_int8):
                print('support int8')
            if (builder.fp16_mode):
                print('fp16 kernels are permitted')
            builder.fp16_mode = True
            #builder.int8_mode = True
            builder.strict_type_constraints = True
            builder.max_workspace_size = 1 << 29 # 512MB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 416, 416]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
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


# Try to load a previously generated yolo network graph in ONNX format:
# However, on this machine, there is an error if using the onnx model directly
# Run ONNX Simplifier <https://github.com/daquexian/onnx-simplifier> with the following command
# python3 -m onnxsim input_onnx_model.onnx output_onnx_model.onnx
onnx_file_path = './trt_models/multi_objs/FRC2020models_v11_powercell_powerport.onnx'
#onnx_file_path = './trt_models/multi_objs/FRC2020models_v11_powercell_powerport_simplified.onnx'
engine_file_path = './trt_models/multi_objs/FRC2020models_v11_powercell_powerport.trt'
input_image_path = './hatchPanel_sample.jpg'


def preprosess_img(img_path):
    frame = cv2.imread(img_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yolo_img =cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
    plt.imshow(img)
    return yolo_img


def main():
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

    plt.imshow(img)

    ## output_shapes = [(1,20,13,13)]
    #trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
    ## num_classes = 4
    ## num_anchors = 5
    ## num_label_points = 19

    ## trt_outputs = array(trt_outputs).reshape(1, num_anchors*(num_label_points + num_classes),13,13)
    # print('trt_outputs type', type(trt_outputs))

    ## print('trt outputs shape ', trt_outputs.shape)


if __name__ == '__main__':
    main()


