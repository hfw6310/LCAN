import argparse
import cv2
import glob
import numpy as np
import os
import torch

from basicsr.archs.lkdn_arch import LKDN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'experiments/LKDN_C55_A55_BSConvU_adan_ema_x4_DF2K111_1000k_111991/models/net_g_990000.pth'  # noqa: E501
    )
    parser.add_argument(
        '--input', type=str, default='/home/hfw/projects/python/LKDN/test_LCAN/Set5/LR_bicubic1/X4', help='input test image folder')
    parser.add_argument('--output', type=str, default='results/LKDN', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = LKDN(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=55,
        num_atten=55,
        num_block=8,
        upscale=4,
        num_in=4,
        conv='BSConvU_ks',
        upsampler='pixelshuffledirect')
    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx + 1, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)        
                start.record(stream=torch.cuda.current_stream())

                output = model(img)

                end.record(stream=torch.cuda.current_stream())
                end.synchronize()
                elapsed_time = start.elapsed_time(end)
                print("elapsed_time:",elapsed_time)

        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}.png'), output)


if __name__ == '__main__':
    main()
