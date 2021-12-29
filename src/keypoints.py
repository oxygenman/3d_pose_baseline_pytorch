#openPose 18 keypoints https://blog.csdn.net/ssyy5233225/article/details/105265488
import json
import io
import torch
import numpy as np
import os
import argparse
from model import LinearModel, weight_init
from torch.autograd import Variable
from data_process import unNormalizeData
import matplotlib.gridspec as gridspec
from camera import *
import matplotlib.pyplot as plt
from vis import *
openPose16={0:"Nose",
            1:"Neck",
            2:"RShoulder",
            3:"RElbow",
            4:"RWrist",
            5:"LShoulder",
            6:"LElbow",
            7:"LWrist",
            8:"RHip",
            9:"RKnee",
            10:"RAnkle",
            11:"LHip",
            12:"LKnee",
            13:"LAnkle",
            14:"REye",
            15:"LEye",
            16:"REar",
            17:"LEar"}

                            #hm36
openPose25={0: "Nose",     #15
            1:"Neck",      #13
            2:"RShoulder", #25
            3:"RElbow",    #26
            4:"RWrist",    #27
            5:"LShoulder", #17
            6:"LElbow",    #18
            7:"LWrist",    #19
            8:"MidHip",    #0
            9:"RHip",      #1
            10:"RKnee",    #2
            11:"RAnkle",   #3
            12:"LHip",     #6
            13:"LKnee",    #7
            14:"LAnkle",   #8
            15:"REye",
            16:"LEye",
            17:"REar",
            18:"LEar",
            19:"LBigToe",
            20:"LSmallToe",
            21:"LHeel",
            22:"RBigToe",
            23:"RSmallToe",
            24:"RHeel"}

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16     #openpose25
SH_NAMES[0]  = 'RFoot' #11
SH_NAMES[1]  = 'RKnee' #10
SH_NAMES[2]  = 'RHip'  #9
SH_NAMES[3]  = 'LHip'  #12
SH_NAMES[4]  = 'LKnee' #13
SH_NAMES[5]  = 'LFoot' #14
SH_NAMES[6]  = 'Hip'   #8
SH_NAMES[7]  = 'Spine' #无
SH_NAMES[8]  = 'Thorax'#1
SH_NAMES[9]  = 'Head'  #0
SH_NAMES[10] = 'RWrist'#4
SH_NAMES[11] = 'RElbow'#3
SH_NAMES[12] = 'RShoulder'#2
SH_NAMES[13] = 'LShoulder'#5
SH_NAMES[14] = 'LElbow'#6
SH_NAMES[15] = 'LWrist'#7


# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip' # 8
H36M_NAMES[1]  = 'RHip'# 9
H36M_NAMES[2]  = 'RKnee'#10
H36M_NAMES[3]  = 'RFoot'#11
H36M_NAMES[6]  = 'LHip' #12
H36M_NAMES[7]  = 'LKnee'#13
H36M_NAMES[8]  = 'LFoot'#14
H36M_NAMES[12] = 'Spine'#无
H36M_NAMES[13] = 'Thorax'#1
H36M_NAMES[14] = 'Neck/Nose'#0
H36M_NAMES[15] = 'Head'#无
H36M_NAMES[17] = 'LShoulder'#5
H36M_NAMES[18] = 'LElbow'#6
H36M_NAMES[19] = 'LWrist'#7
H36M_NAMES[25] = 'RShoulder'#2
H36M_NAMES[26] = 'RElbow'#6
H36M_NAMES[27] = 'RWrist'#4
#将openpose25关键点映射到17个关键点
#dim_to_use = (0,1,2,3,6,7,8,12,13,15,17,18,19,25,26,27)
dim_to_use= np.array([0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15, 16, 17, 24, 25, 26,
       27, 30, 31, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53, 54, 55])


#标准化2d数据
def normalize_data(hm36_data,data_mean,data_std,dim_to_use):
    hm36_data=hm36_data[dim_to_use]
    mu=data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    normalized = np.divide((hm36_data-mu),stddev)
    return normalized

def read_keypoints(json_input):
    with io.open(json_input, encoding='utf-8') as f:
        keypoint_dicts = json.loads(f.read())["people"]   
        pose_pts = (np.array(keypoint_dicts[0]["pose_keypoints_2d"]).reshape(25, 3))[:,:-1]
    return pose_pts
def covert_op_to_hm(op_kp_2d):
    order=[15,13,25,26,27,17,18,19,0,1,2,3,6,7,8]
    hm36_item=np.zeros([32,2])
    for i in range(len(order)):
        hm36_item[order[i]]=op_kp_2d[i]
    #set spin
    hm36_item[12]=(op_kp_2d[1]+op_kp_2d[8])/2
    hm36_item=hm36_item.flatten()
    return hm36_item


    
def parse_arg():
    parser = argparse.ArgumentParser(description="test car")
    parser.add_argument('--checkpoint', help='resume checkpoint .pth', type=str,default="./checkpoint/example/ckpt_best.pth.tar")
    parser.add_argument('--test_dir', help='test dir', type=str,default="./data")
    args = parser.parse_args()
    print(args)
    return args

if __name__== "__main__":
    args=parse_arg()
    device ="cuda:0" if torch.cuda.is_available() else "cpu:0"    
    stat_3d = torch.load(os.path.join("./data", 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join("./data", 'stat_2d.pth.tar'),encoding="latin1")

    pose_pts=read_keypoints("./input_2d/frame002144_keypoints.json")
    hm36_pts=covert_op_to_hm(pose_pts)
    print("hm36_pts.shape:",hm36_pts.shape)
    n_hm36_pts=normalize_data(hm36_pts,stat_2d["mean"],stat_2d["std"],dim_to_use)
    inps=n_hm36_pts
    print("n_hm36_psts.shape:",n_hm36_pts.shape)

    model = LinearModel()
    #load weight
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.to(device)
    inputs = Variable(torch.tensor(inps.reshape(1,32)).to(torch.float32).to(device))
    outputs = model(inputs)
    p3d = unNormalizeData(outputs.cpu().detach().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
    p2d = unNormalizeData(inps.reshape(1,32), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
    rcams = load_cameras()
    R, T, f, c, k, p, name=rcams[(9,2)]
    p3d = cam2world_centered(p3d,R,T)
    fig = plt.figure( figsize=(6.4, 3.2) )
    gs1 = gridspec.GridSpec(1, 3) # 5 rows, 9 columns
    gs1.update(wspace=-0.00, hspace=0.05) # set the spacing between axes.
    plt.axis('on')
    #show 2d
    ax1 = plt.subplot(gs1[0])
    show2Dpose(p2d, ax1)
    ax1.invert_yaxis()
    #show pred 3d
    ax3 = plt.subplot(gs1[2], projection='3d')
    p3d = p3d[0,:]
    show3Dpose(p3d, ax3,lcolor="#9b59b6", rcolor="#2ecc71")
    plt.savefig("openpose.png")
    plt.close()