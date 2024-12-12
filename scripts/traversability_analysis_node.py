#!/usr/bin/env python3
import rospy
import topological_mapping
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import sys
from pathlib import Path
from topological_mapping.srv import (
    TraversabilityAnalyzer,
    TraversabilityAnalyzerRequest,
    TraversabilityAnalyzerResponse,
)
import cv2
import cv_bridge


from topological_mapping.learning_trav.models import DINOv2TraversabilityAnalyser


def pixel_to_one_hot(pixel_x, pixel_y, depth, img):
    dino_patch_size = 14
    img_size = 224

    dino_res = img_size // dino_patch_size
    one_hot = torch.zeros((dino_res, dino_res), dtype=torch.float32)
    # Aspect ratio check, and check axis ordering is correct
    assert img.shape[1] > img.shape[0]

    scale_x = float(dino_res) / float(img.shape[1])
    scale_y = float(dino_res) / float(img.shape[0])
    coord_x = int(scale_x * pixel_x)
    coord_y = int(scale_y * pixel_y)
    one_hot[coord_x, coord_y] = depth
    return one_hot.reshape((dino_res**2))


def trav_handler(device, model, bridge, req: TraversabilityAnalyzerRequest):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    starter.record()
    transform = ToTensor()
    imgA_cv2 = bridge.imgmsg_to_cv2(req.nodeA)
    # imgB = bridge.imgmsg_to_cv2(req.nodeB)
    imgA = transform(cv2.resize(imgA_cv2, (224, 224)))
    imgA = imgA.unsqueeze(dim=0)
    # FIXME will me ignored for now as I don't deal with bi-directional edges
    # for now.
    # imgB = transform(cv2.resize(imgB, (224, 224)))
    T = torch.tensor([req.TBtoA.x, req.TBtoA.y, req.TBtoA.z])
    T = T.to(device)
    imgA = imgA.to(device)
    T = T.unsqueeze(dim=0)
    pixel_onehot = pixel_to_one_hot(
        req.pixel_x.data, req.pixel_y.data, req.TBtoA.z, imgA_cv2
    )
    pixel_onehot = pixel_onehot.to(device).unsqueeze(dim=0)
    # model_out = F.softmax(model(imgA, None, T), dim=-1)
    model_out = F.softmax(model(imgA, pixel_onehot), dim=-1)
    model_out = torch.argmax(model_out)
    ender.record()
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    rospy.loginfo(f"Inference time: {curr_time}")
    out = TraversabilityAnalyzerResponse()
    out.response = model_out
    if model_out.item() == 0:
        rospy.loginfo("Request is traversable")
    elif model_out.item() == 1:
        rospy.loginfo("Request is untraversable")
    else:
        raise ValueError()
    return out


if __name__ == "__main__":
    rospy.init_node("traversability_analyser_node")
    bridge = cv_bridge.CvBridge()
    device = torch.device("cuda")
    torch.cuda.init()
    myargv = rospy.myargv(sys.argv)
    model: DINOv2TraversabilityAnalyser = torch.load(open(Path(myargv[1]), "rb"))
    model.to(device)
    trav_handler_closure = lambda req: trav_handler(device, model, bridge, req)
    service = rospy.Service(
        "traversability_analyser",
        TraversabilityAnalyzer,
        handler=trav_handler_closure,
    )
    rospy.spin()
