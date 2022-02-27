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
import curriculums_con as curriculums
from random import randrange
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from torchvision.utils import save_image
import copy

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


checkpoint = torch.load('/scratch_net/biwidl212/shecai/models/pi-gan_pretrain/srnchairs/checkpoint_train_srnchairs.pth', map_location=torch.device(device))
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
curriculum['num_steps'] = 96
curriculum['last_back'] = True
curriculum['img_size'] = 128
curriculum['psi'] = 1
curriculum['nerf_noise'] = 0

curriculum_opt = curriculums.extract_metadata(getattr(curriculums, 'srnchairs'), 300000)
curriculum_opt['num_steps'] = 24
curriculum_opt['last_back'] = True
curriculum_opt['img_size'] = 64
curriculum_opt['psi'] = 1
curriculum_opt['nerf_noise'] = 0

resize64 = transforms.Compose([transforms.Resize((64, 64), interpolation=0)])
inv_normalize = transforms.Normalize(
    mean=[-0.5/0.5],
    std=[1/0.5]
    )
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
        dpath = data["path"][0]
        obj_basename = os.path.basename(dpath)
        obj_name = obj_basename
        if has_output and obj_name in finished:
            print("(skip)")
            continue
        images = [data["images"][0][64]]
        for i in range(NV-1):
            images.append(data["images"][0][randrange(250)])
        images = torch.stack(images, 0)  # (NV, 3, H, W)

        # NV, _, H, W = images.shape
        _, _, H, W = images.shape

        if all_rays is None:

            NS = len(source)
            src_view_mask = torch.zeros(NV, dtype=torch.bool)
            src_view_mask[source] = 1

            target_view_mask = target_view_mask_init.clone()
            target_view_mask *= ~src_view_mask

            novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)

        n_gen_views = len(novel_view_idxs)
        z = torch.rand((10000, 256), device=device) * 2 - 1
        with torch.no_grad():
            frequencies, phase_shifts = generator.siren.mapping_network(z)
        w_frequencies_global = frequencies.mean(0, keepdim=True)
        w_phase_shifts_global = phase_shifts.mean(0, keepdim=True)
        _, pos = encoder(resize64(images[src_view_mask]).to(device=device), 1)
        # frequencies, phase_shifts = generator.siren.mapping_network(z)
        # images = []
        with torch.enable_grad():
            w_frequencies = frequencies.mean(0, keepdim=True)
            w_phase_shifts = phase_shifts.mean(0, keepdim=True)

            w_frequency_offsets = torch.zeros_like(w_frequencies)
            w_phase_shift_offsets = torch.zeros_like(w_phase_shifts)
            w_frequency_offsets.requires_grad_()
            w_phase_shift_offsets.requires_grad_()

            # z.requires_grad_()
            optimizer = torch.optim.Adam([w_frequency_offsets, w_phase_shift_offsets], lr=1e-2, weight_decay = 1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.75)

            n_iterations = 100

            for i in range(n_iterations):
                # frame, _ = generator.forward(z, pos[:, 0], pos[:, 1], mode='recon', max_batch_size=opt.max_batch_size, **curriculum)
                noise_w_frequencies = 0.003 * torch.randn_like(w_frequencies) * (n_iterations - i)/n_iterations
                noise_w_phase_shifts = 0.003 * torch.randn_like(w_phase_shifts) * (n_iterations - i)/n_iterations
                frame, _ = generator.forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, pos[:, 0], pos[:, 1], mode='recon', **curriculum_opt)
                loss = torch.nn.MSELoss()(frame, resize64(images[src_view_mask]).to(device))
                loss = loss.mean()
                # print(loss)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if i % 10 == 0:
                    with torch.no_grad():
                        obj_out_dir = os.path.join(output_dir, obj_name)
                        os.makedirs(obj_out_dir, exist_ok=True)
                        out_file = os.path.join(
                            obj_out_dir, f"{i}.jpg")
                        img, _ = generator.staged_forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, pos[:, 0], pos[:, 1], mode='recon', lock_view_dependence=True, **curriculum)
                        save_image(inv_normalize(img), out_file, normalize=False)
                        # for pitch in [-0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7]:
                        for yaw in [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]:
                            
                            copied_metadata = copy.deepcopy(curriculum)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] = yaw
                            copied_metadata['v_mean'] = pos[:, 0]
                            # img, _ = generator.staged_forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, pitch, yaw, max_batch_size=94800000, mode='recon', lock_view_dependence=True, **curriculum)
                            img, _ = generator.staged_forward_with_frequencies(w_frequencies + noise_w_frequencies + w_frequency_offsets, w_phase_shifts + noise_w_phase_shifts + w_phase_shift_offsets, None, None, max_batch_size=94800000, lock_view_dependence=True, **copied_metadata)
                            # save_image(img, f"debug/{i}_{pitch}_{yaw}.jpg", normalize=True)
                            obj_out_dir = os.path.join(output_dir, obj_name)
                            os.makedirs(obj_out_dir, exist_ok=True)
                            # for i in range(n_gen_views):
                            out_file = os.path.join(
                                obj_out_dir, f"{i}_{yaw}.jpg")

                            # imageio.imwrite(out_file, (img * 255).astype(np.uint8))
                            save_image(inv_normalize(img), out_file, normalize=False)
        _, img_pos = encoder(resize64(images[target_view_mask]).to(device=device), 1)
        # img_z = img_z.repeat(NV, 1)
        # frequencies = frequencies.repeat(NV, 1)
        # phase_shifts = phase_shifts.repeat(NV, 1)
        final_frequencies = w_frequencies + w_frequency_offsets
        final_phase_shift = w_phase_shifts + w_phase_shift_offsets
        final_frequencies = final_frequencies.repeat(NV, 1)
        final_phase_shift = final_phase_shift.repeat(NV, 1)
        all_rgb = []
        for split in range(NV-1):
            # subset_z = img_z[split * 1:(split+1) * 1]
            subset_frequencies = final_frequencies[split * 1:(split+1) * 1]
            subset_phase_shift = final_phase_shift[split * 1:(split+1) * 1]
            subset_pos = img_pos[split * 1:(split+1) * 1]
            # g_imgs, _ = generator(subset_z, subset_pos[:, 0], subset_pos[:, 1], mode='recon', **curriculum)
            g_imgs, _ = generator.staged_forward_with_frequencies(subset_frequencies, subset_phase_shift, subset_pos[:, 0], subset_pos[:, 1], max_batch_size=94800000, mode='recon', lock_view_dependence=True, **curriculum)

            all_rgb.append(g_imgs)

        all_rgb = torch.cat(all_rgb, axis=0)
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
                # ssim = cal_ssim(
                #     all_rgb[view_idx],
                #     rgb_gt_all[view_idx],
                #     multichannel=True,
                #     data_range=1,
                # )
                # psnr = cal_psnr(
                #     all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                # )
                # curr_ssim += ssim
                # curr_psnr += psnr

                if args.write_compare:
                    out_file = os.path.join(
                        obj_out_dir,
                        "{:06}_compare.png".format(novel_view_idxs[view_idx].item()),
                    )
                    out_im = np.hstack((all_rgb[view_idx], rgb_gt_all[view_idx]))
                    imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
#         curr_psnr /= n_gen_views
#         curr_ssim /= n_gen_views
#         curr_cnt = 1
#         total_psnr += curr_psnr
#         total_ssim += curr_ssim
#         cnt += curr_cnt
#         if not args.no_compare_gt:
#             print(
#                 "curr psnr",
#                 curr_psnr,
#                 "ssim",
#                 curr_ssim,
#                 "running psnr",
#                 total_psnr / cnt,
#                 "running ssim",
#                 total_ssim / cnt,
#                 flush=True
#             )
#         finish_file.write(
#             "{} {} {} {}\n".format(obj_name, curr_psnr, curr_ssim, curr_cnt)
#         )
# print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt, flush=True)
