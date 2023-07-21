import sys
import os
import argparse
from multiprocessing import Process, Queue
from typing import List, Dict
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", "-g", type=str, required=True,
                            help="space delimited GPU id list (global id in nvidia-smi, "
                                 "not considering CUDA_VISIBLE_DEVICES)")
parser.add_argument('--eval', action='store_true', default=False,
                   help='evaluation mode (run the render_imgs script)')
parser.add_argument('--dataset', type=str, default='syn', choices=['syn', 'llff','tt','nsvf'])
parser.add_argument("--render_path", type=int, default=0)
parser.add_argument("--render_only", type=int, default=0)
parser.add_argument("--suffix", type=str, default='v0')
parser.add_argument("-f","--force", action="store_true")

args = parser.parse_args()

PSNR_FILE_NAME = 'test_psnr.txt'

def run_exp(env,  config, datadir, expname, basedir, ckpt=None, suffix='v0'):
    base_cmd = ['python', 'vectquant.py', '--autotask',  '--config', config, '--datadir', datadir, '--expname', expname, '--basedir', basedir]
    base_cmd = base_cmd + ['--render_path', str(args.render_path)]
    base_cmd = base_cmd + ['--render_only', str(args.render_only)]
    if ckpt is not None:
        base_cmd = base_cmd + ['--ckpt', ckpt]
    base_cmd = base_cmd + ['--suffix', suffix]
    # psnr_file_path = os.path.join(basedir, expname,'imgs_test_all','test_res.txt')
    psnr_file_path = os.path.join(basedir, expname+'_'+suffix+'_vq','extreme_load','extreme_load_res.txt')
    if os.path.isfile(psnr_file_path) and not args.force:
        print('! SKIP', psnr_file_path, "on ", env["CUDA_VISIBLE_DEVICES"])
        return
    print('********************************************')
    opt_cmd = ' '.join(base_cmd)
    print(opt_cmd, "on ", env["CUDA_VISIBLE_DEVICES"])
    opt_ret = subprocess.check_output(opt_cmd, shell=True, env=env).decode(
        sys.stdout.encoding)


def process_main(device, queue):
    # Set CUDA_VISIBLE_DEVICES programmatically
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    while True:
        task = queue.get()
        if len(task) == 0:
            break
      
        run_exp(env, **task)


DatasetSetting={
    "syn": {
        "data": "/bfs/HoloResearch/NeRFData/nerf_synthetic",
        "cfg": f"configs/vq/syn.txt",
        "scene_list":[ 'chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship'],
        "basedir":f"./log_reimp/syn"
    },
    "llff":{
        "data": "/bfs/HoloResearch/NeRFData/nerf_llff_data",
        "cfg": "configs/vq/llff.txt",
        "scene_list": ['fern', 'flower', 'room', 'leaves', 'horns', 'trex', 'fortress', 'orchids'],
        "basedir": "./log_reimp/llff"
    },
    "tt":{
        "data": "/bfs/HoloResearch/NeRFData/TanksAndTemple",
        "cfg": "configs/vq/tt.txt",
        "scene_list": ['Barn','Caterpillar','Family','Ignatius', 'Truck'],
        "basedir": "./log_reimp/tt"
    },
    "nsvf":{
        "data": "/bfs/HoloResearch/NeRFData/Synthetic_NSVF",
        "cfg": "configs/vq/nsvf.txt",
        "scene_list": ['Bike','Lifestyle','Palace','Robot','Spaceship','Steamtrain','Toad','Wineholder'],
        "basedir": "./log_reimp/nsvf"
    }
}


datasetting = DatasetSetting[args.dataset]
all_tasks = []
for scene in datasetting["scene_list"]:
    task: Dict = {}
    task['datadir'] = f'{datasetting["data"]}/{scene}'
    task['expname'] = f'{scene}'  
    task["config"] = datasetting['cfg']
    task["basedir"] = datasetting["basedir"]
    task["ckpt"] = f"{datasetting['basedir']}/{scene}/{scene}.th"
    task["suffix"] = args.suffix
    assert os.path.exists(task['datadir']), task['datadir'] + ' does not exist'
    assert os.path.isfile(task['config']), task['config'] + ' does not exist'
    all_tasks.append(task)

pqueue = Queue()
for task in all_tasks:
    pqueue.put(task)

args.gpus = list(map(int, args.gpus.split()))
print('GPUS:', args.gpus)

for _ in args.gpus:
    pqueue.put({})

all_procs = []
for i, gpu in enumerate(args.gpus):
    process = Process(target=process_main, args=(gpu, pqueue))
    process.daemon = True
    process.start()
    all_procs.append(process)

for i, gpu in enumerate(args.gpus):
    all_procs[i].join()



class AverageMeter(object):
    def __init__(self, name=''):
        self.name=name
        self.reset()
    def reset(self):
        self.val=0
        self.sum=0
        self.avg=0
        self.count=0
    def update(self,val,n=1):
        self.val=val
        self.sum += val*n
        self.count += n
        self.avg=self.sum/self.count
    def __repr__(self) -> str:
        return f'{self.name}: average {self.count}: {self.avg}\n'

from prettytable import PrettyTable
table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS_ALEX","LPIPS_VGG","SIZE"])
table.float_format = '.4'

PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS_A=AverageMeter('LPIPS_A')
LPIPS_V=AverageMeter('LPIPS_V')
SIZE=AverageMeter('SIZE')
for scene in datasetting['scene_list']:
    path = f'./{datasetting["basedir"]}/{scene}/imgs_test_all/test_res.txt'
    if not os.path.exists(path):
        path = f'./{datasetting["basedir"]}/{scene}/imgs_test_all/testmean.txt'

    with open(path, 'r') as f:
        lines = f.readlines()
        psnr = float(lines[0].strip())
        ssim = float(lines[1].strip())
        lpips_a = float(lines[2].strip())
        lpips_v = float(lines[3].strip())
        PSNR.update(psnr)
        SSIM.update(ssim)
        LPIPS_A.update(lpips_a)
        LPIPS_V.update(lpips_v)
        uncompressed_file = f'./{datasetting["basedir"]}/{scene}/{scene}.th'
        if os.path.exists(uncompressed_file):
            size = os.path.getsize(uncompressed_file)/(1024*1024)
        else:
            size = 0
        SIZE.update(size)
        table.add_row([scene, psnr, ssim, lpips_a, lpips_v,size])
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS_A.avg,LPIPS_V.avg, SIZE.avg])

writedir = os.path.join(datasetting["basedir"], 'merge.txt')
with open(writedir, 'w') as f:
    f.writelines(table.get_string())
print(table)


table = PrettyTable(["Scene", "PSNR", "SSIM", "LPIPS_ALEX","LPIPS_VGG","SIZE"])
table.float_format = '.4'

PSNR=AverageMeter('PSNR')
SSIM=AverageMeter('SSIM')
LPIPS_A=AverageMeter('LPIPS_A')
LPIPS_V=AverageMeter('LPIPS_V')
SIZE=AverageMeter('SIZE')
for scene in datasetting['scene_list']:
    path = f'./{datasetting["basedir"]}/{scene}_{args.suffix}_vq/extreme_load/extreme_load_res.txt'
    if not os.path.exists(path):
        path = f'./{datasetting["basedir"]}/{scene}_{args.suffix}_vq/test5/vq_quant_0.7_res.txt'

    with open(path, 'r') as f:
        lines = f.readlines()
        psnr = float(lines[0].strip())
        ssim = float(lines[1].strip())
        lpips_a = float(lines[2].strip())
        lpips_v = float(lines[3].strip())
        PSNR.update(psnr)
        SSIM.update(ssim)
        LPIPS_A.update(lpips_a)
        LPIPS_V.update(lpips_v)
        uncompressed_file = f'./{datasetting["basedir"]}/{scene}_{args.suffix}_vq/extreme_ckpt.pt'
        compressed_file = f'./{datasetting["basedir"]}/{scene}_{args.suffix}_vq/extreme_ckpt.zip'
        if os.path.exists(compressed_file):
            size = os.path.getsize(compressed_file)/(1024*1024)
        else:
            size = 0
        SIZE.update(size)
        table.add_row([scene, psnr, ssim, lpips_a, lpips_v,size])
table.add_row(['Mean', PSNR.avg, SSIM.avg, LPIPS_A.avg,LPIPS_V.avg, SIZE.avg])

writedir = os.path.join(datasetting["basedir"], f'merge_{args.suffix}_vq.txt')
with open(writedir, 'w') as f:
    f.writelines(table.get_string())
print(table)