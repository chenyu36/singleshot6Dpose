import os
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.misc
import warnings
warnings.filterwarnings("ignore")

from darknet_multi import Darknet
from utils import *
import dataset_multi
from MeshPly import MeshPly

check_z_plausibility = False
debug_multi_boxes = True

def valid(datacfg0, datacfg1, datacfg2, datacfg3, cfgfile, weightfile, conf_th):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    vertices = []
    corners3D = []
    # Parse configuration file 0
    options       = read_data_cfg(datacfg0)
    valid_images  = options['valid']
    meshname      = options['mesh']
    name          = options['name']
    prefix        = 'results'
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices.append(np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose())
    corners3D.append(get_3D_corners(vertices[0]))
    diam          = float(options['diam'])


    # Parse configuration file 1
    options       = read_data_cfg(datacfg1)
    valid_images  = options['valid']
    meshname      = options['mesh']
    name          = options['name']
    prefix        = 'results'
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices.append(np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose())
    corners3D.append(get_3D_corners(vertices[1]))
    diam          = float(options['diam'])

    # Parse configuration file 2
    options       = read_data_cfg(datacfg2)
    valid_images  = options['valid']
    meshname      = options['mesh']
    name          = options['name']
    prefix        = 'results'
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices.append(np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose())
    corners3D.append(get_3D_corners(vertices[2]))
    diam          = float(options['diam'])

    # Parse configuration file 3
    options       = read_data_cfg(datacfg3)
    valid_images  = options['valid']
    meshname      = options['mesh']
    name          = options['name']
    prefix        = 'results'
    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices.append(np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose())
    corners3D.append(get_3D_corners(vertices[3]))
    diam          = float(options['diam'])

    #define the paths to tensorRT models 
    onnx_file_path = './trt_models/multi_objs/FRC2020models_v6_powerCell_retrained_simplified.onnx'
    engine_file_path = './trt_models/multi_objs/FRC2020models_v6_powerCell_retrained_simplified.trt'
    # onnx_file_path = './trt_models/multi_objs/FRC2020models_v5_tilt_camera_simplified.onnx'
    # engine_file_path = './trt_models/multi_objs/FRC2020models_v5_tilt_camera_simplified.trt'
    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()
    dist = get_camera_distortion_mat()

    # Get validation file names
    with open(valid_images) as fp:
        tmp_files = fp.readlines()
        valid_files = [item.rstrip() for item in tmp_files]
    
    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    # comment out since we are loading TRT model using get_engine() function
    # model = Darknet(cfgfile)
    # model.load_weights(weightfile)
    # model.cuda()
    # model.eval()
    model_input_size = [416, 416]
    test_width = 416
    test_height = 416

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Specify the number of workers for multiple processing, get the dataloader for the test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True}


    # Parameters
    use_cuda        = True
    # add here
    num_classes     = 4
    anchors         = [1.4820, 2.2412, 2.0501, 3.1265, 2.3946, 4.6891, 3.1018, 3.9910, 3.4879, 5.8851] 
    num_anchors     = 5
    eps             = 1e-5
    conf_thresh     = conf_th
    iou_thresh      = 0.5 # was 0.5
    nms_thresh      = 0.8 # was 0.5
    y_dispay_thresh = 1 # was 144

    # Parameters to save
    errs_2d             = []
    edges = [[1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 4], [3, 7], [4, 8], [5, 6], [5, 7], [6, 8], [7, 8]]
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    # Iterate through test batches (Batch size for test data is 1)
    #logging('Testing {}...'.format(name))
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while cap.isOpened():
            retflag, frame = cap.read() 
            if retflag:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.undistort(img, internal_calibration, dist, None, internal_calibration)
                yolo_img =cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)

                all_boxes = []
                output = []
                #all_boxes = do_detect_multi_v3(model, yolo_img, conf_thresh, nms_thresh)

                #detection_result = do_detect_multi_v2(model, yolo_img, conf_thresh, nms_thresh)
                all_boxes = do_detect_trt_multi(context, yolo_img, conf_thresh, nms_thresh, num_classes, anchors, num_anchors, bindings, inputs, outputs, stream)
                
                # for i in range(num_classes):
                #     correspondingclass = i
                #     # experiment: chris
                #     # override the default confidence threshold
                #     conf_thresh = 0.20
                #     boxes = get_corresponding_region_boxes(detection_result, conf_thresh, model.num_classes, model.anchors, model.num_anchors, correspondingclass, only_objectness=1, validation=True)[0]
                #     boxes = nms_multi_v2(boxes, nms_thresh)
                #     all_boxes.append(boxes)
        

                # debug: print all the detected box's class
                for box in all_boxes:
                    if debug_multi_boxes:
                        print('box\n', box)
                    #print('box cluster')
                    for i in range(len(box)):
                        print('box class ', int(box[i][20]), ' confidence ', "{0:.2f}".format(float(box[i][18])))     

                # all_boxes = nms_multi(all_boxes, nms_thresh)
                
                for boxes in all_boxes:

                    # For each image, get all the predictions

                    #boxes   = all_boxes[i][0]
                    #correspondingclass = i + 1            
                    best_conf_est = -1
                    

                    # If the prediction has the highest confidence, choose it as our prediction

                        # if (boxes[18] > best_conf_est) and (boxes[20] == correspondingclass):
                        #     best_conf_est = boxes[18]
                        #     box_pr        = boxes
                        #     bb2d_pr       = get_2d_bb(box_pr[:18], output.size(3))
                    #print('checking class ', boxes[20])
                    for i in range(num_classes):
                        correspondingclass = i
                        # skipping brownGlyph which is class 0
                        if correspondingclass == 0:
                            continue
                        for j in range(len(boxes)):
     
                            if (boxes[j][20] == correspondingclass):
                                print('detected obj class is ', correspondingclass)
                                box_pr        = boxes[j]

                                # Denormalize the corner predictions 
                                corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
                                
                                corners2D_pr[:, 0] = corners2D_pr[:, 0] * 1280
                                corners2D_pr[:, 1] = corners2D_pr[:, 1] * 720
                                # Compute [R|t] by pnp
                                # print('corners3D \n', corners3D)

                                objpoints3D = np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[i][:3, :]), axis=1)), dtype='float32')

                                # the order of 3D vertices from the above function is incorrect, for upperPortRed and upperPortBlue, 
                                # so we manually calculate 3D points (i.e. centroid + vertices) manually
                                # the order of vertices is according to this link (see point 2)
                                # https://github.com/microsoft/singleshotpose/blob/master/label_file_creation.md
                                if (correspondingclass == 2 or correspondingclass == 3):
                                    x_min_3d = 0
                                    x_max_3d = 1.2192
                                    y_min_3d = 0
                                    y_max_3d = 1.1176
                                    z_min_3d = 0
                                    z_max_3d = 0.003302
                                    centroid = [(x_min_3d+x_max_3d)/2, (y_min_3d+y_max_3d)/2, (z_min_3d+z_max_3d)/2]

                                    objpoints3D = np.array([centroid,\
                                    [ x_min_3d, y_min_3d, z_min_3d],\
                                    [ x_min_3d, y_min_3d, z_max_3d],\
                                    [ x_min_3d, y_max_3d, z_min_3d],\
                                    [ x_min_3d, y_max_3d, z_max_3d],\
                                    [ x_max_3d, y_min_3d, z_min_3d],\
                                    [ x_max_3d, y_min_3d, z_max_3d],\
                                    [ x_max_3d, y_max_3d, z_min_3d],\
                                    [ x_max_3d, y_max_3d, z_max_3d]])
                                elif (correspondingclass == 1):
                                    x_min_3d = 0
                                    x_max_3d = 0.1778
                                    y_min_3d = 0
                                    y_max_3d = 0.1778
                                    z_min_3d = 0
                                    z_max_3d = 0.1778
                                    centroid = [(x_min_3d+x_max_3d)/2, (y_min_3d+y_max_3d)/2, (z_min_3d+z_max_3d)/2]

                                    objpoints3D = np.array([centroid,\
                                    [ x_min_3d, y_min_3d, z_min_3d],\
                                    [ x_min_3d, y_min_3d, z_max_3d],\
                                    [ x_min_3d, y_max_3d, z_min_3d],\
                                    [ x_min_3d, y_max_3d, z_max_3d],\
                                    [ x_max_3d, y_min_3d, z_min_3d],\
                                    [ x_max_3d, y_min_3d, z_max_3d],\
                                    [ x_max_3d, y_max_3d, z_min_3d],\
                                    [ x_max_3d, y_max_3d, z_max_3d]])
                                elif (correspondingclass == 0):
                                    x_min_3d = 0
                                    x_max_3d = 0.1524
                                    y_min_3d = 0
                                    y_max_3d = 0.1524
                                    z_min_3d = 0
                                    z_max_3d = 0.1524
                                    centroid = [(x_min_3d+x_max_3d)/2, (y_min_3d+y_max_3d)/2, (z_min_3d+z_max_3d)/2]

                                    objpoints3D = np.array([centroid,\
                                    [ x_min_3d, y_min_3d, z_min_3d],\
                                    [ x_min_3d, y_min_3d, z_max_3d],\
                                    [ x_min_3d, y_max_3d, z_min_3d],\
                                    [ x_min_3d, y_max_3d, z_max_3d],\
                                    [ x_max_3d, y_min_3d, z_min_3d],\
                                    [ x_max_3d, y_min_3d, z_max_3d],\
                                    [ x_max_3d, y_max_3d, z_min_3d],\
                                    [ x_max_3d, y_max_3d, z_max_3d]])

                                # troubleshooting rvecs
                                # print('objpoints3D \n', objpoints3D)
                                # print('corners2D_pr \n', corners2D_pr)
                                K = np.array(internal_calibration, dtype='float32')

                                rvec, R_pr, t_pr = pnp(objpoints3D,  corners2D_pr, K)
                            
                                # Compute pixel error

                                # Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)

                                # proj_2d_pred = compute_projection(vertices[i], Rt_pr, internal_calibration) 

                                # proj_corners_pr = np.transpose(compute_projection(corners3D[i], Rt_pr, internal_calibration))

                                draw_bbox_for_obj(corners2D_pr, t_pr, frame, y_dispay_thresh)

                                # only draw axis if the object is the upper power port
                                if (correspondingclass == 2 or correspondingclass == 3):
                                    frame = draw_axis(rvec, t_pr, frame, corners2D_pr)

                cv2.imshow('6D pose estimation - multi-objects', frame)
                detectedKey = cv2.waitKey(1) & 0xFF
                if detectedKey == ord('c'):
                    timestamp = time.time()
                    cv2.imwrite('./screenshots/screeshot' + str(timestamp) + '.jpg', frame)
                    print('captured screeshot')
                elif detectedKey == ord('q'):
                    print('quitting program')
                    break

def are_corners_greater_than_y_thres(corners, y_thresh):
    for corner in corners:
        x = corner[0]
        y = corner[1]
        if y <= y_thresh:
            return False
    return True

def find_pt_with_smallest_y(pts):
    ymin = 10000
    ymin_pt = (0, 0)
    # get the x of center point which has the index 0
    x = pts[0][0]
    for pt in pts:
        y = pt[1]
        if y < ymin:
            ymin = y
            ymin_pt = (x, pt[1])
    return ymin_pt

def draw_cube(img, pts):
    if len(pts)==9:
        
        thickness = 2
        # rear face
        r = (0,0,255)
        g = (0,255,0)
        b = (255,0,0)
        color = r
        cv2.line(img,pts[2],pts[4],color,thickness)
        cv2.line(img,pts[4],pts[8],color,thickness)
        cv2.line(img,pts[8],pts[6],color,thickness)
        cv2.line(img,pts[6],pts[2],color,thickness)
        # left face
        color = b
        cv2.line(img,pts[1],pts[2],color,thickness)
        cv2.line(img,pts[3],pts[4],color,thickness)
        # right face
        cv2.line(img,pts[5],pts[6],color,thickness)
        cv2.line(img,pts[7],pts[8],color,thickness)
        # front face
        color = g
        cv2.line(img,pts[1],pts[5],color,thickness)
        cv2.line(img,pts[5],pts[7],color,thickness)
        cv2.line(img,pts[7],pts[3],color,thickness)
        cv2.line(img,pts[3],pts[1],color,thickness)    

def draw_bbox_for_obj(corners2D_pr, t_pr, frame, y_dispay_thresh):

    corner2d_pr_vertices = []
    index = 0
    # not an empty array, AND, all corners are beyond a y threshold
    if (corners2D_pr.size > 0) and are_corners_greater_than_y_thres(corners2D_pr, y_dispay_thresh):
        ymin_pt = find_pt_with_smallest_y(corners2D_pr)
        pt_for_label1= (int(ymin_pt[0]-30), int(ymin_pt[1]-30))
        pt_for_label2 = (int(ymin_pt[0]-50), int(ymin_pt[1]-10))
        for pt in corners2D_pr:
            # print('corners2D_pr', pt)
            x = pt[0]
            y = pt[1]
            pt =(x, y)
            if y > y_dispay_thresh:
                white = (255, 255, 255)
                cv2.circle(frame, pt, 2, white, thickness=2, lineType=8, shift=0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 255, 255)
                font_scale = 0.6
                pt_for_number = (int(x+5), int(y-5))
                # only print the center point (index 0)
                if index == 0:
                    cv2.putText(frame, str(index), pt_for_number, font, font_scale, color, 2, lineType=8)
                # skip the centroid, we only want the vertices
                corner2d_pr_vertices.append(pt)

                
            index = index + 1
        blue = (255,0,0)

        white = (255,255,255)
        
        # print x offset and z offset (depth) above the smallest y point
        x1 = pt_for_label1[0]
        y1 = pt_for_label1[1]
        purple = (132,37,78)
        #cv2.rectangle(frame, (x1, y1-20), (x1+len(x_cord)*19+60,y1), purple, -1)
        

        z = float(t_pr[2])

        # z is valid only if it's negative (i.e. away from camera)
        if check_z_plausibility:
            if (z <= 0):
                z_cord = 'Depth ' + str("{0:.2f}".format(z)) + 'm'

                x = float(t_pr[0])
                x_cord = 'x ' + str("{0:.2f}".format(x)) + 'm'
            
                x2 = pt_for_label2[0]
                y2 = pt_for_label2[1]                        
                cv2.rectangle(frame, (x2-5, y2-20*2), (x2+len(z_cord)*12,y2+5), purple, -1)
                cv2.putText(frame, x_cord, pt_for_label1, font, font_scale, white, 1, lineType=8)
                cv2.putText(frame, z_cord, pt_for_label2, font, font_scale, white, 1, lineType=8)
        else:
            z_cord = 'Depth ' + str("{0:.2f}".format(z)) + 'm'

            x = float(t_pr[0])
            x_cord = 'x ' + str("{0:.2f}".format(x)) + 'm'
            
            x2 = pt_for_label2[0]
            y2 = pt_for_label2[1]                        
            cv2.rectangle(frame, (x2-5, y2-20*2), (x2+len(z_cord)*12,y2+5), purple, -1)
            cv2.putText(frame, x_cord, pt_for_label1, font, font_scale, white, 1, lineType=8)
            cv2.putText(frame, z_cord, pt_for_label2, font, font_scale, white, 1, lineType=8)

        draw_cube(frame, corner2d_pr_vertices)
        # if z is less than zero; i.e. away from camera
        if (t_pr[2] < 0):
            print('x ', round(float(t_pr[0]), 2), 'y ', round(float(t_pr[1]), 2), 'z ', round(float(t_pr[2]), 2)) 


def draw_axis(rvecs, tvecs, frame, corners2D_pr):
    # Virtual World points of trihedron to show target pose
    size = 0.5
    axis = np.array([[0,0,0],[size,0,0],[0,size,0],[0,0,size]], dtype=np.float32)

    internal_calibration = get_camera_intrinsic()
    K = np.array(internal_calibration, dtype='float32')
    dist = get_camera_distortion_mat()

    print('rvecs \n', rvecs*180/3.142857)

    axisPoints, _ = cv2.projectPoints(axis, rvecs, tvecs, K, dist)

    # axis origin working at lower left corner of the
    # frame = cv2.line(frame, tuple(axisPoints[0].ravel()), tuple(axisPoints[1].ravel()), (255,0,0), 3)
    # frame = cv2.line(frame, tuple(axisPoints[0].ravel()), tuple(axisPoints[2].ravel()), (0,255,0), 3)
    # frame = cv2.line(frame, tuple(axisPoints[0].ravel()), tuple(axisPoints[3].ravel()), (0,0,255), 3)

    # calculate the delta (from the centroid in 2d image to the axisPoint origin)
    delta = corners2D_pr[0].ravel() - axisPoints[0].ravel()
    # draw axis from the centroid of object, and apply delta to each axis (i.e. x or y or z)
    frame = cv2.line(frame, tuple(corners2D_pr[0].ravel()), tuple(axisPoints[1].ravel() + delta), (255,0,0), 3)
    frame = cv2.line(frame, tuple(corners2D_pr[0].ravel()), tuple(axisPoints[2].ravel() + delta), (0,255,0), 3)
    frame = cv2.line(frame, tuple(corners2D_pr[0].ravel()), tuple(axisPoints[3].ravel() + delta), (0,0,255), 3)
    return frame


if __name__ == '__main__' and __package__ is None:
    import sys
    print(sys.argv)
    if len(sys.argv) == 3:
        conf_th = 0.45
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]

        #class number = 0
        datacfg0 = 'multi_obj_pose_estimation/cfg/brownGlyph.data'
        #class number = 1
        datacfg1 = 'multi_obj_pose_estimation/cfg/powerCell_occlusion.data'
        #class number = 2
        datacfg2 = 'multi_obj_pose_estimation/cfg/upperPortRed.data'
        #class number = 3
        datacfg3 = 'multi_obj_pose_estimation/cfg/upperPortBlue_occlusion.data'
        
        valid(datacfg0, datacfg1, datacfg2, datacfg3, cfgfile, weightfile, conf_th)

    else:
        print('Usage:')
        print(' python valid.py cfgfile weightfile')
