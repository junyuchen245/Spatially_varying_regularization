from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph_Cond import CONFIGS as CONFIGS_TM
import models.TransMorph_Cond as TransMorph
import torch.nn.functional as F
from scipy.ndimage.interpolation import map_coordinates, zoom
from surface_distance import compute_dice_coefficient
import scipy
import digital_diffeomorphism as dd

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    atlas_dir = 'D:/DATA/IXI_brain/network_training/atlas.pkl'
    val_dir = 'D:/DATA/IXI_brain/network_training/Val/'
    weights = [1, 4, 5]  # loss weights 0.02
    save_dir = 'TransMorphCondOMT_ncc_{}_OMTCond_{}_localDiff_{}/'.format(weights[0], weights[1], weights[2])
    csv_name = save_dir[:-1]
    csv_writter(csv_name+',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,', csv_name)
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    max_epoch = 500 #max traning epoch
    cont_training = False #if continue training
    sigmas = np.linspace(0.1, 1.0, num=6)
    #sigmas = (sigmas ** 3) * sigma_max
    print(sigmas)
    time_steps = 4
    '''
    Initialize model
    '''
    H, W, D = 160, 192, 224
    config = CONFIGS_TM['TransMorph-3-LVL']
    config.img_size = (H // 2, W // 2, D // 2)
    config.window_size = (H // 32, W // 32, D // 32)
    model = TransMorph.TransMorphCascadeAd(config, SVF=False)
    model_dir = 'experiments/' + save_dir
    model_idx = -1
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    spatial_trans = TransMorph.SpatialTransformer((H, W, D))
    spatial_trans.cuda()
    '''
    Initialize training
    '''
    val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                       trans.NumpyType((np.float32, np.int16))])
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    for reg_i in np.arange(0, 5., 0.05):
        eval_dsc = utils.AverageMeter()
        line = '{}'.format(reg_i)
        line_log_det = '{}'.format(reg_i)
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                x = data[0]
                y = data[1]
                x = F.avg_pool3d(x, 2).cuda()
                y = F.avg_pool3d(y, 2).cuda()
                x_seg = data[2].cuda()
                y_seg = data[3].cuda()
                reg_code = torch.tensor([reg_i/weights[1]], dtype=x.dtype, device=x.device).unsqueeze(dim=0)
                output, wts = model((x, y), reg_code)
                flow = F.interpolate(output.cuda(), scale_factor=2, mode='trilinear') * 2
                x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
                x_seg_oh = torch.squeeze(x_seg_oh, 1)
                x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
                # x_segs = model.spatial_trans(x_seg.float(), flow.float())
                x_segs = []
                for i in range(46):
                    def_seg = spatial_trans(x_seg_oh[:, i:i + 1, ...].float(), flow.float())
                    x_segs.append(def_seg)
                x_segs = torch.cat(x_segs, dim=1)
                def_out = torch.argmax(x_segs, dim=1, keepdim=True)
                del x_segs, x_seg_oh
                disp_field = flow.cpu().detach().numpy()[0]
                disp_field = np.array([zoom(disp_field[i], 0.5, order=2) for i in range(3)]).astype(np.float16).astype('float32')
                disp_field = np.array([zoom(disp_field[i], 2, order=2) for i in range(3)])
                jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
                log_jac_det = np.log(jac_det).std()
                #mask = x_seg.cpu().detach().numpy()[0, 0, 1:-1, 1:-1, 1:-1]
                #mask = mask > 0
                #trans_ = disp_field + dd.get_identity_grid(disp_field)
                #jac_dets = dd.calc_jac_dets(trans_)
                #non_diff_voxels, non_diff_tetrahedra, non_diff_volume = dd.calc_measurements(jac_dets, mask)
                #total_voxels = np.sum(mask)
                #log_jac_det = non_diff_volume / total_voxels * 100
                dice = utils.dice_val_VOI(def_out.long(), y_seg.long())
                dice = dice.item()
                eval_dsc.update(dice, x.size(0))
                print(eval_dsc.avg)
                line += ',{}'.format(dice)
                line_log_det += ',{}'.format(log_jac_det)
        csv_writter(line+','+line_log_det, csv_name)

def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (
                jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
             jacobian[1, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :,
                                                                                                       :, :]) + \
             jacobian[2, 0, :, :, :] * (
                         jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :,
                                                                                                       :, :])

    return jacdet

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    torch.manual_seed(0)
    main()