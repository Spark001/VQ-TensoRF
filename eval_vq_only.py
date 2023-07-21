import os
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
import datetime

from dataLoader import dataset_dict
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


def eval_vq(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    assert os.path.isfile(args.ckpt), f'No checkpoint found at {args.ckpt}'

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    tensorf = TensorVMSplitVQ(**kwargs)
    tensorf.extreme_load(ckpt)

    from functools import partial
    evaluation_test = partial(evaluation,test_dataset=test_dataset, args=args, 
                            renderer=renderer,white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                            compute_extra_metrics=args.autotask, im_save=args.autotask)
    logfolder = os.path.dirname(args.ckpt)
    os.makedirs(f'{logfolder}/extreme_load', exist_ok=True)
    TestLoad = evaluation_test(tensorf=tensorf, N_vis=-1, savePath=f"{logfolder}/extreme_load" if not args.debug else None)
    print(f'======>Test ExtremeLoad: AfterVQFinetune WithFinetune(Quant) {args.expname} test all psnr: {np.mean(TestLoad)} <========================')


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    eval_vq(args)
