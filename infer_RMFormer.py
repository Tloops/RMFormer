import glob
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.RMFormer import CONFIGS as CONFIGS_TM
import models.RMFormer as RMFormer
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from PIL import Image
import cv2
from rmse_utils import point_spatial_transformer

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[:, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def field_visualizer(field_numpy, imtype=np.float32):
    '''
    field_numpy: (h, w, 2)
    put field_numpy into a 3-channel image using the red and green channels
    the blue channel is set to 0
    '''
    nh, nw, _ = field_numpy.shape
    tmp = np.zeros((nh, nw, 3))
    tmp[:, :, :2] = field_numpy
    field_np = tmp
    field_np -= np.amin(field_np)
    field_np /= np.amax(field_np)
    field_np = field_np * 255
    return field_np.astype(imtype)

def main():
    test_dir = './FIRE'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'RMFormer_ssim_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder

    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if not os.path.exists('Quantitative_Results/' + model_folder):
        os.makedirs('Quantitative_Results/' + model_folder)
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    line = ',SSIM,det'
    csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

    config = CONFIGS_TM['TransMorph-No-Conv-Skip']
    model = RMFormer.RMFormer(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location='cuda:0')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    test_set = datasets.FIREInferDataset(test_dir, size=256, transforms=None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    ssim = SSIM(data_range=255, size_average=True, channel=1)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        total_rmse = 0
        total_rmse_raw = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x_rgb = data[0]
            y_rgb = data[1]
            x = data[2]
            y = data[3]
            cps = data[4][0]

            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], x)
            eval_dsc_def.update(ncc.item(), x.numel())
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())
            line = 'p{}'.format(stdy_idx) + ',' + str(ncc.item()) + ',' + str(np.sum(jac_det <= 0) / np.prod(x.shape))
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            # stdy_idx += 1
            # flip image
            x_in = torch.cat((x, y), dim=1)
            output = model(x_in)
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], y)
            eval_dsc_def.update(ncc.item(), y.numel())
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :])
            line = 'p{}'.format(stdy_idx) + ',' + str(ncc.item()) + ',' + str(np.sum(jac_det <= 0) / np.prod(x.shape))
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

            # result image saving
            x_origin = x.squeeze().detach().cpu().numpy()
            y_origin = y.squeeze().detach().cpu().numpy()
            pred_img = output[0].squeeze().detach().cpu().numpy()
            save_image(x_origin, 'Quantitative_Results/' + model_folder[:-1] + '/p{}_x.png'.format(stdy_idx))
            save_image(y_origin, 'Quantitative_Results/' + model_folder[:-1] + '/p{}_y.png'.format(stdy_idx))
            save_image(pred_img, 'Quantitative_Results/' + model_folder[:-1] + '/p{}_pred.png'.format(stdy_idx))

            # control point RMSE
            print(output[1].shape)
            flow = output[1].squeeze().permute(1, 2, 0)
            fix_point = data[4][:, :, 2:]
            mov_point = data[4][:, :, :2]

            viz_field = field_visualizer(flow.detach().cpu().numpy())
            cv2.imwrite('Quantitative_Results/' + model_folder[:-1] + '/p{}_field.png'.format(stdy_idx), viz_field)

            print("flow.shape", flow.shape)
            print("fix_point.shape", fix_point.shape)

            data = [t.cuda() for t in [mov_point, flow]]
            warp_point = point_spatial_transformer(data)
            print("warp_point", warp_point)
            print("warp_point.shape", warp_point.shape)

            mse = torch.sum((warp_point - fix_point) ** 2)
            mse_raw = torch.sum((fix_point - mov_point) ** 2)
            print(mse)

            # compute grid
            grid_img = mk_grid_img(8, 1, (x.shape[0], config.img_size[0], config.img_size[1]))
            def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
            def_gridimg = def_grid.detach().cpu().numpy()[0, 0, :, :]*255
            save_image(def_gridimg, 'Quantitative_Results/' + model_folder[:-1] + '/p{}_grid.png'.format(stdy_idx))


            # control point RMSE origin implementation
            # mse = 0
            # mse_raw = 0
            # img_x = cv2.imread('Quantitative_Results/' + model_folder[:-1] + '/p{}_x.png'.format(stdy_idx))
            # img_y = cv2.imread('Quantitative_Results/' + model_folder[:-1] + '/p{}_y.png'.format(stdy_idx))
            # img_pred = cv2.imread('Quantitative_Results/' + model_folder[:-1] + '/p{}_pred.png'.format(stdy_idx))
            # for i in range(len(cps)):
            #     orix, oriy, dstx, dsty = cps[i][0], cps[i][1], cps[i][2], cps[i][3]
            #     cv2.circle(img_x, (int(orix), int(oriy)), 1, (0, 0, 255), 2)
            #     cv2.circle(img_y, (int(dstx), int(dsty)), 1, (0, 0, 255), 2)
            #     prdx = orix + output[1][0][0][int(torch.round(orix))][int(torch.round(oriy))]
            #     prdy = oriy + output[1][0][1][int(torch.round(orix))][int(torch.round(oriy))]
            #     cv2.circle(img_pred, (int(prdx), int(prdy)), 1, (0, 0, 255), 2)
            #     mse += (prdx - dstx) ** 2 + (prdy - dsty) ** 2
            #     mse_raw += (orix - dstx) ** 2 + (oriy - dsty) ** 2
            # cv2.imwrite('Quantitative_Results/' + model_folder[:-1] + '/p{}_x_p.png'.format(stdy_idx), img_x)
            # cv2.imwrite('Quantitative_Results/' + model_folder[:-1] + '/p{}_y_p.png'.format(stdy_idx), img_y)
            # cv2.imwrite('Quantitative_Results/' + model_folder[:-1] + '/p{}_pred_p.png'.format(stdy_idx), img_pred)
            rmse = torch.sqrt(mse / len(cps))
            rmse_raw = torch.sqrt(mse_raw / len(cps))
            total_rmse += rmse
            total_rmse_raw += rmse_raw
            
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        print('Deformed RMSE: {:.3f}, Affine RMSE: {:.3f}'.format(total_rmse / len(test_set),
                                                                  total_rmse_raw / len(test_set)))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    # GPU_iden = 5
    # GPU_num = torch.cuda.device_count()
    # print('Number of GPU: ' + str(GPU_num))
    # for GPU_idx in range(GPU_num):
    #     GPU_name = torch.cuda.get_device_name(GPU_idx)
    #     print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    # torch.cuda.set_device(GPU_iden)
    # print(GPU_iden)
    # GPU_avai = torch.cuda.is_available()
    # print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    # print('If the GPU is available? ' + str(GPU_avai))
    main()