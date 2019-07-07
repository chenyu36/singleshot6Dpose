import os
import cv2
import pickle as pkl
import time
import torch
import random
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
import time
warnings.filterwarnings("ignore")


from darknet import Darknet
import dataset
from utils import *
from MeshPly import MeshPly
from imgutil import process_result, load_images, resize_image, cv_image2tensor, transform_result
from cvtorchvision import cvtransforms
from numpy import array
import pycuda.driver as cuda
import pycuda.autoinit

def cv_image2tensor(img, size):
    img = resize_image(img, size)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float() / 255.0

    return img

# resize_image by scaling while preserving aspect ratio and then padding remaining area with gray pixels
def resize_image(img, size):
    h, w = img.shape[0:2]
    newh, neww = size
    scale = min(newh / h, neww / w)
    img_h, img_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((newh, neww, 3), 128.0)
    canvas[(newh - img_h) // 2 : (newh - img_h) // 2 + img_h, (neww - img_w) // 2 : (neww-img_w) // 2 + img_w, :] = img

    return canvas

def draw_bbox(imgs, bbox, colors, classes):
    img = imgs[int(bbox[0])]
    #label = classes[int(bbox[-1])]
    label = 'cargo'
    p1 = tuple(bbox[1:3].int())
    p2 = tuple(bbox[3:5].int())
    color = random.choice(colors)
    cv2.rectangle(img, p1, p2, color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(imgget_region_boxes, p3, p4, color, -1)
    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def detect_video(model, args):

    model_input_size = [int(model.net_info['height']), int(model.net_info['width'])]

    colors = pkl.load(open("pallete", "rb"))
    classes = 'cargo'
    colors = [colors[1]]
    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')
    while cap.isOpened():
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, model_input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh)
            if len(detections) != 0:
                detections = transform_result(detections, [frame], model_input_size)
                classes = ['cargo']
                for detection in detections:
                    draw_bbox([frame], detection, colors, classes)

            if not args.no_show:
                cv2.imshow('frame', frame)
            out.write(frame)
            if read_frames % 30 == 0:
                print('Number of frames processed:', read_frames)
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    end_time = datetime.now()
    print('Detection finished in %s' % (end_time - start_time))
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    return

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

# Create new directory
def makedirs(path):
    if not os.path.exists( path ):
        os.makedirs( path )

def are_corners_greater_than_y_thres(corners, y_thresh):
    for corner in corners:
        x = corner[0]
        y = corner[1]
        if y <= y_thresh:
            return False
    return True

def valid(datacfg, cfgfile, weightfile, outfile):
    def truths_length(truths):
        for i in range(50):
            if truths[i][1] == 0:
                return i

    # Parse configuration files
    options      = read_data_cfg(datacfg)
    valid_images = options['valid']
    meshname     = options['mesh']
    backupdir    = options['backup']
    name         = options['name']
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    # Parameters
    prefix       = 'results'
    seed         = int(time.time())
    gpus         = '0'     # Specify which gpus to use
    test_width   = 544
    test_height  = 544
    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)
    save            = False
    testtime        = True
    use_cuda        = True
    num_classes     = 1
    testing_samples = 0.0
    eps             = 1e-5
    notpredicted    = 0 
    conf_thresh     = 0.1
    nms_thresh      = 0.5 # was 0.4
    match_thresh    = 0.5
    y_dispay_thresh = 144
    # Try to load a previously generated yolo network graph in ONNX format:
    #onnx_file_path = './cargo_yolo2.onnx'
    #engine_file_path = './cargo_yolo2.trt'
    #onnx_file_path = './cargo_yolo2_c920_cam.onnx'
    #engine_file_path = './cargo_yolo2_c920_cam.trt'
    onnx_file_path = './cargo_yolo2_c920_cam_83percent.onnx'
    engine_file_path = './cargo_yolo2_c920_cam_83percent.trt'

    if save:
        makedirs(backupdir + '/test')
        makedirs(backupdir + '/test/pr')

    # To save
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []
    errs_corner2D       = []
    preds_trans         = []
    preds_rot           = []
    preds_corners2D     = []

    # Read object model information, get 3D bounding box corners
    mesh          = MeshPly(meshname)
    vertices      = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    print('vertices', vertices)
    corners3D     = get_3D_corners(vertices)
    print('corners3D', corners3D)
    # diam          = calc_pts_diameter(np.array(mesh.vertices))
    diam          = float(options['diam'])

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
    # model.print_network()
    # model.load_weights(weightfile)
    # model.cuda()
    # model.eval()
    model_input_size = [416, 416]
    # print('model.anchors', model.anchors)
    # print('model.num_anchors', model.num_anchors)

    # specify the webcam as camera
    colors = pkl.load(open("pallete", "rb"))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    #transform = transforms.Compose([transforms.ToTensor(),])

    transform = cvtransforms.Compose([
 
         cvtransforms.Resize(size=(416, 416), interpolation='BILINEAR'),
 
         cvtransforms.ToTensor()
 
         ])

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        while cap.isOpened():
            retflag, frame = cap.read() 
            if retflag:
                #resize_frame = cv2.resize(frame, (416, 416), interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.undistort(img, internal_calibration, dist, None, internal_calibration)
                yolo_img =cv2.resize(img, (416, 416), interpolation=cv2.INTER_AREA)
                box_pr_multi = do_detect_trt(context, yolo_img, conf_thresh, nms_thresh, bindings, inputs, outputs, stream)

                for box_pr in box_pr_multi:
                    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')           
                    corners2D_pr[:, 0] = corners2D_pr[:, 0] * 1280
                    corners2D_pr[:, 1] = corners2D_pr[:, 1] * 720
                    preds_corners2D.append(corners2D_pr)
                    
                    # Compute [R|t] by pnp
                    R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)), dtype='float32'),  corners2D_pr, np.array(internal_calibration, dtype='float32'))

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
                        # print x offset and z offset (depth) above the smallest y point
                        x = float(t_pr[0])
                        x_cord = 'x ' + str("{0:.2f}".format(x)) + 'm'
                        cv2.putText(frame, x_cord, pt_for_label1, font, font_scale, blue, 2, lineType=8)
                        z = float(t_pr[2])
                        z_cord = 'Depth ' + str("{0:.2f}".format(z)) + 'm'
                        cv2.putText(frame, z_cord, pt_for_label2, font, font_scale, blue, 2, lineType=8) 
                        draw_cube(frame, corner2d_pr_vertices)
                        # if z is less than zero; i.e. away from camera
                        if (t_pr[2] < 0):
                            print('x ', round(float(t_pr[0]), 2), 'y ', round(float(t_pr[1]), 2), 'z ', round(float(t_pr[2]), 2))

                if save:
                    preds_trans.append(t_pr)
                    preds_rot.append(R_pr)

                    np.savetxt(backupdir + '/test/pr/R_' + valid_files[count][-8:-3] + 'txt', np.array(R_pr, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr/t_' + valid_files[count][-8:-3] + 'txt', np.array(t_pr, dtype='float32'))
                    np.savetxt(backupdir + '/test/pr/corners_' + valid_files[count][-8:-3] + 'txt', np.array(corners2D_pr, dtype='float32'))


                    # Compute 3D distances
                    transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                    vertex_dist       = np.mean(norm3d)


                cv2.imshow('6D pose estimation', frame)
                detectedKey = cv2.waitKey(1) & 0xFF
                if detectedKey == ord('c'):
                    timestamp = time.time()
                    cv2.imwrite('./screenshots/screeshot' + str(timestamp) + '.jpg', frame)
                    print('captured screeshot')
                elif detectedKey == ord('q'):
                    print('quitting program')
                    break

                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                t5 = time.time()
            else:
                break



            if False:
                print('-----------------------------------')
                print('  tensor to cuda : %f' % (t2 - t1))
                print('         predict : %f' % (t3 - t2))
                print('get_region_boxes : %f' % (t4 - t3))
                print('            eval : %f' % (t5 - t4))
                print('           total : %f' % (t5 - t1))
                print('-----------------------------------')


    if save:
        predfile = backupdir + '/predictions_linemod_' + name +  '.mat'
        scipy.io.savemat(predfile, {'R_prs': preds_rot, 't_prs':preds_trans, 'corner_prs': preds_corners2D})

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 4:
        datacfg = sys.argv[1]
        cfgfile = sys.argv[2]
        weightfile = sys.argv[3]
        outfile = 'comp4_det_test_'
        valid(datacfg, cfgfile, weightfile, outfile)
    else:
        print('Usage:')
        print(' python valid.py datacfg cfgfile weightfile')
