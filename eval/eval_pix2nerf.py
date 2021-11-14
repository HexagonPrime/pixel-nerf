"""
Full evaluation script, including PSNR+SSIM evaluation with multi-GPU support.

python eval.py --gpu_id=<gpu list> -n srn_chairs -c srn.conf -D /scratch_net/biwidl212/shecai/srn_chairs/chairs -F srn
python eval/eval.py -D /scratch_net/biwidl212/shecai/srn_chairs/chairs -n srn_chairs -P '64' -O /scratch_net/biwidl212/shecai/eval_out/srn_chair_1v
"""
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import numpy as np
import imageio
import skimage.measure
import util
from data import get_split_dataset
from model import make_model
from render import NeRFRenderer
import cv2
import tqdm
import ipdb
import warnings

from torch_ema import ExponentialMovingAverage
import torchvision.transforms as transforms
#  from pytorch_memlab import set_target_gpu
#  set_target_gpu(9)
import curriculums_con as curriculums
from random import randrange
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr

def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="0",
        help="Source view(s) for each object.",
    )
    parser.add_argument(
        "--no_compare_gt",
        action="store_true",
        help="Skip GT comparison (metric won't be computed) and only render images",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="eval",
        help="If specified, saves generated images to directory",
    )

    parser.add_argument(
        "--write_compare", action="store_true", help="Write GT comparison image"
    )
    return parser


args, conf = util.args.parse_args(
    extra_args, default_expname="shapenet",
)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)
data_loader = torch.utils.data.DataLoader(
    dset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False
)

output_dir = args.output.strip()
has_output = len(output_dir) > 0

total_psnr = 0.0
total_ssim = 0.0
cnt = 0

if has_output:
    finish_path = os.path.join(output_dir, "finish.txt")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(finish_path):
        with open(finish_path, "r") as f:
            lines = [x.strip().split() for x in f.readlines()]
        lines = [x for x in lines if len(x) == 4]
        finished = set([x[0] for x in lines])
        total_psnr = sum((float(x[1]) for x in lines))
        total_ssim = sum((float(x[2]) for x in lines))
        cnt = sum((int(x[3]) for x in lines))
        if cnt > 0:
            print("resume psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
        else:
            total_psnr = 0.0
            total_ssim = 0.0
    else:
        finished = set()

    finish_file = open(finish_path, "a", buffering=1)
    print("Writing images to", output_dir)


checkpoint = torch.load('/scratch_net/biwidl212/shecai/srnchairs_progressive_rl5_ssim1_vgg1_plg0_el1_ep1_cl1_pl0_fe1/checkpoint_train.pth', map_location=torch.device(device))
generator = checkpoint['generator.pth'].to(device)
encoder = checkpoint['encoder.pth'].to(device)
ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
ema_encoder = ExponentialMovingAverage(encoder.parameters(), decay=0.999)
ema.load_state_dict(checkpoint['ema.pth'])
ema_encoder.load_state_dict(checkpoint['ema_encoder.pth'])
ema.copy_to(generator.parameters())
ema_encoder.copy_to(encoder.parameters())
generator.set_device(device)
generator.eval()
# encoder.set_device(device)
encoder.eval()

source = torch.tensor(sorted(list(map(int, args.source.split()))), dtype=torch.long)

NV = 4

target_view_mask = torch.ones(NV, dtype=torch.bool)
target_view_mask_init = target_view_mask

all_rays = None
rays_spl = []

src_view_mask = None
total_objs = len(data_loader)

curriculum = curriculums.extract_metadata(getattr(curriculums, 'srnchairs'), 300000)
curriculum['num_steps'] = 64
# curriculum['psi'] = 0.7
curriculum['last_back'] = True
curriculum['img_size'] = 128
curriculum['psi'] = 1
# curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['nerf_noise'] = 0

resize64 = transforms.Compose([transforms.Resize((64, 64), interpolation=0)])
with torch.no_grad():
    for obj_idx, data in enumerate(data_loader):
        print(
            "OBJECT",
            obj_idx,
            "OF",
            total_objs,
            "PROGRESS",
            obj_idx / total_objs * 100.0,
            "%",
            data["path"][0],
        )
        # dpath = [data["path"][0][64]]
        # for i in range(3):
        #     dpath = dpath + data["path"][0][randrange(250)]
        # dpath = torch.cat(dpath, 0)
        dpath = data["path"][0]
        obj_basename = os.path.basename(dpath)
        obj_name = obj_basename
        if has_output and obj_name in finished:
            print("(skip)")
            continue
        images = [data["images"][0][64]]
        for i in range(3):
            images.append(data["images"][0][randrange(250)])
        images = torch.stack(images, 0)  # (NV, 3, H, W)

        # NV, _, H, W = images.shape
        _, _, H, W = images.shape

        if all_rays is None:

            NS = len(source)
            src_view_mask = torch.zeros(NV, dtype=torch.bool)
            src_view_mask[source] = 1

            # poses = data["poses"][0]  # (NV, 4, 4)
            # src_poses = poses[src_view_mask].to(device=device)  # (NS, 4, 4)

            target_view_mask = target_view_mask_init.clone()
            target_view_mask *= ~src_view_mask

            novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)

            # focal = focal.to(device=device)

        # rays_spl = torch.split(all_rays, args.ray_batch_size, dim=0)  # Creates views

        n_gen_views = len(novel_view_idxs)
        img_z, _ = encoder(resize64(images[src_view_mask]).to(device=device), 1)
        _, img_pos = encoder(resize64(images[target_view_mask]).to(device=device), 1)
        img_z = img_z.repeat(NV, 1)
        all_rgb = []
        # gen_positions = []
        for split in range(NV-1):
            subset_z = img_z[split * 1:(split+1) * 1]
            subset_pos = img_pos[split * 1:(split+1) * 1]
            g_imgs, _ = generator(subset_z, subset_pos[:, 0], subset_pos[:, 1], mode='recon', **curriculum)

            all_rgb.append(g_imgs)
            # gen_positions.append(g_pos)

        all_rgb = torch.cat(all_rgb, axis=0)
        # gen_positions = torch.cat(gen_positions, axis=0)

        # all_rgb = []
        # for rays in tqdm.tqdm(rays_spl):
        #     rgb = render_par(rays[None])
        #     rgb = rgb[0].cpu()
        # all_rgb = generator(img_z, )
        # all_rgb, _ = generator(img_z, img_pos[:, 0], img_pos[:, 1], mode='recon', **curriculum)
        print(all_rgb.min())
        print(all_rgb.max())
        print(all_rgb.shape)

        # all_rgb = torch.cat(all_rgb, dim=0)
        all_rgb = all_rgb * 0.5 + 0.5
        all_rgb = torch.clamp(
            all_rgb.permute(0, 2, 3, 1), 0.0, 1.0
        ).cpu().numpy()  # (NV-NS, H, W, 3)
        if has_output:
            obj_out_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_out_dir, exist_ok=True)
            for i in range(n_gen_views):
                out_file = os.path.join(
                    obj_out_dir, "{:06}.png".format(novel_view_idxs[i].item())
                )
                imageio.imwrite(out_file, (all_rgb[i] * 255).astype(np.uint8))

        curr_ssim = 0.0
        curr_psnr = 0.0
        if not args.no_compare_gt:
            images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
            images_gt = images_0to1[target_view_mask]
            rgb_gt_all = (
                images_gt.permute(0, 2, 3, 1).contiguous().numpy()
            )  # (NV-NS, H, W, 3)
            for view_idx in range(n_gen_views):
                # ssim = skimage.measure.compare_ssim(
                #     all_rgb[view_idx],
                #     rgb_gt_all[view_idx],
                #     multichannel=True,
                #     data_range=1,
                # )
                ssim = cal_ssim(
                    all_rgb[view_idx],
                    rgb_gt_all[view_idx],
                    multichannel=True,
                    data_range=1,
                )
                # psnr = skimage.measure.compare_psnr(
                #     all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                # )
                psnr = cal_psnr(
                    all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                )
                curr_ssim += ssim
                curr_psnr += psnr

                if args.write_compare:
                    out_file = os.path.join(
                        obj_out_dir,
                        "{:06}_compare.png".format(novel_view_idxs[view_idx].item()),
                    )
                    out_im = np.hstack((all_rgb[view_idx], rgb_gt_all[view_idx]))
                    imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
        curr_psnr /= n_gen_views
        curr_ssim /= n_gen_views
        curr_cnt = 1
        total_psnr += curr_psnr
        total_ssim += curr_ssim
        cnt += curr_cnt
        if not args.no_compare_gt:
            print(
                "curr psnr",
                curr_psnr,
                "ssim",
                curr_ssim,
                "running psnr",
                total_psnr / cnt,
                "running ssim",
                total_ssim / cnt,
            )
        finish_file.write(
            "{} {} {} {}\n".format(obj_name, curr_psnr, curr_ssim, curr_cnt)
        )
print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
