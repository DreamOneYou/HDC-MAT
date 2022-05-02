# -*- coding: utf-8 -*-
# 这份测试代码主要实现了对图像分块预测。因为我的网络结构不支持全图（240,240,155），因此需要分块（128,128,128）大小输入网络。具体代码如下：
from time import time
import os
import math
import argparse
import warnings
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from data import datasets
import models
import logging
from utils import Parser,str2bool
from torch.utils.data import DataLoader
root_path = '/home/icml/wpx/brain/2018_valid_pkl'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-cfg', '--cfg', default='HDC_Net', required=False, type=str,
                        help='Your detailed configuration of the network')
    parser.add_argument('--mode', default=None,
                        help='')
    parser.add_argument('-gpu', '--gpu', default='0', type=str)
    parser.add_argument('--is_out', default=True, type=bool)
    parser.add_argument('--patchsize', default=(128, 128, 128), type=int, metavar='N',
                        help='number of slice')
    parser.add_argument('-restore', '--restore', default=argparse.SUPPRESS, type=str,
                        help='The path to restore the model.')  # 'model_epoch_300.pth'
    args = parser.parse_args()
    args = Parser(args.cfg, log='train').add_args(args)
    args.gpu = str(args.gpu)
    ckpts = args.makedir()
    args.resume = os.path.join(ckpts, args.restore)  # specify the epoch

    return args


def generate_test_locations(image, patch_size, stride):
    # 40-128-160, liver_patch shape
    bs, C, ww, hh, dd = image.shape
    # print('image.shape', bs, C, ww, hh, dd)
    sz = math.ceil((ww - patch_size[2]) / stride[0]) + 1
    sx = math.ceil((hh - patch_size[1]) / stride[1]) + 1
    sy = math.ceil((dd - patch_size[2]) / stride[2]) + 1

    return (sz, sx, sy), (bs, C, ww, hh, dd)


def infer_tumorandliver(model, ct_array_nor, file_index, names, out_dir, original=(240, 240, 155), cube_shape=(128, 128, 128),
                        use_TTA=False, postprocess=True,verbose=True):
    patch_size = cube_shape
    patch_stride = [32, 32, 32]
    # patch_stride = [64, 160, 160]
    H, W, T = original
    mask_pred_containers = np.zeros((ct_array_nor.shape)).astype(np.float32)  # 用来存放结果
    locations, image_shape = generate_test_locations(ct_array_nor, patch_size, patch_stride)  # 32 64 80
    # print('location', locations, image_shape)

    seg = np.zeros((ct_array_nor.shape)).astype(np.float32)
    cnt = np.zeros((ct_array_nor.shape)).astype(np.float32)
    # print('seg_liver shape', seg.shape)
    start_time = time()
    for z in range(0, locations[0]):
        zs = min(patch_stride[0] * z, image_shape[2] - patch_size[0])
        for x in range(0, locations[1]):
            xs = min(patch_stride[1] * x, image_shape[3] - patch_size[1])
            for y in range(0, locations[2]):
                ys = min(patch_stride[2] * y, image_shape[4] - patch_size[2])
                # print("zs:{},xs:{},ys{}".format(zs, xs, ys))
                patch = ct_array_nor[:, :, zs:zs + patch_size[0],
                        xs:xs + patch_size[1],
                        ys:ys + patch_size[2]]
                # print('patch',patch)
                patch_tensor = torch.from_numpy(patch).cuda()
                if not use_TTA:
                    # torch.cuda.synchronize()
                    logit = model(patch_tensor)
                    output = F.softmax(logit, dim=1)
                else:
                    logit = F.softmax(model(patch_tensor), 1)  # 000
                    logit += F.softmax(model(patch_tensor.flip(dims=(2,))).flip(dims=(2,)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(3,))).flip(dims=(3,)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(4,))).flip(dims=(4,)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(2, 3))).flip(dims=(2, 3)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(2, 4))).flip(dims=(2, 4)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(3, 4))).flip(dims=(3, 4)), 1)
                    logit += F.softmax(model(patch_tensor.flip(dims=(2, 3, 4))).flip(dims=(2, 3, 4)), 1)
                    output = logit / 8.0  # mean
                output = output.cpu().data.numpy()
                # print('output',output.shape)
                seg[:, :, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] = output
                cnt[:, :, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] \
                    = cnt[:, :, zs:zs + patch_size[0], xs:xs + patch_size[1], ys:ys + patch_size[2]] + 1

    output = seg / cnt
    # output = output.astype(np.uint8)
    # print("output:",output.shape)
    output = output[0, :, :H, :W, :T]
    output = output.argmax(0) # (channels,height,width,depth)
    if postprocess == True:
        ET_voxels = (output == 3).sum()
        if ET_voxels < 500:
            output[np.where(output == 3)] = 1

    name = names[file_index]
    oname = os.path.join(out_dir, 'submission', name + '.nii.gz')
    print(oname)
    seg_img = np.zeros(shape=(H, W, T), dtype=np.uint8)

    seg_img[np.where(output == 1)] = 1
    seg_img[np.where(output == 2)] = 2
    seg_img[np.where(output == 3)] = 4
    if verbose:
        print('1:', np.sum(seg_img == 1), ' | 2:', np.sum(seg_img == 2), ' | 4:', np.sum(seg_img == 4))
        print('WT:', np.sum((seg_img == 1) | (seg_img == 2) | (seg_img == 4)), ' | TC:',
              np.sum((seg_img == 1) | (seg_img == 4)), ' | ET:', np.sum(seg_img == 4))
    nib.save(nib.Nifti1Image(seg_img, None), oname)
    end_time = time()
    print("pred:{}, spend time:{}".format(name, end_time-start_time))
def main():
    args = parse_args()

    # create model
    # print("=> creating model %s" %args.arch)
    Network = getattr(models, args.net)  #
    model = Network(**args.net_params)
    model = torch.nn.DataParallel(model).cuda()
    print(args.resume)
    assert os.path.isfile(args.resume), "no checkpoint found at {}".format(args.resume)
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_iter = checkpoint['iter']
    model.load_state_dict(checkpoint['state_dict'])
    msg = ("=> loaded checkpoint '{}' (iter {})".format(args.resume, checkpoint['iter']))
    msg += '\n' + str(args)
    logging.info(msg)
    model.eval()
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    Dataset = getattr(datasets, args.dataset)  #
    valid_list = os.path.join(root_path, args.valid_list)
    valid_set = Dataset(valid_list, root=root_path, for_train=False, transforms=args.test_transforms)  # type tensor, shape x, [X,Y,Z,C]、 y[]
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        collate_fn=valid_set.collate,
        num_workers=10,
        pin_memory=True)
    if args.is_out:
        out_dir = './output/{}'.format(args.cfg)
        os.makedirs(os.path.join(out_dir,'submission'),exist_ok=True)
        os.makedirs(os.path.join(out_dir,'snapshot'),exist_ok=True)
    else:
        out_dir = ''
    start_time = time()
    for file_index, data in enumerate(valid_loader):

        # 将要预测的CT读入
        x, target = data[:2]
        x = x.cpu().data.numpy()
        # 开始预测
        infer_tumorandliver(model, x, file_index, valid_set.names, out_dir, original=(240, 240, 155), cube_shape=(128,128,128))

        torch.cuda.empty_cache()
    end_time = time()
    print("----------------all_time:{}------------".format(end_time - start_time))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()