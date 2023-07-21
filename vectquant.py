
import os
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:


        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)


        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/', prtx='test',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    return f'{logfolder}/{args.expname}.th'

# calculate the importance of each sample by plane and line
def calc_importance(tensorf:TensorVMSplitVQ, args, folder, overwrite=False):
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    path = os.path.join(folder, 'importance.pth')
    if os.path.exists(path) and not overwrite:
        print("Load existed importance file:", path)
        tensorf.importance = torch.load(path)
        return
    pseudo_density_planes = [ torch.ones_like(plane).mean(1,keepdim=True) for plane in tensorf.density_plane]
    for plane in pseudo_density_planes:
        plane.requires_grad = True
    img_eval_interval = 1
    for idx, samples in tqdm(enumerate(train_dataset.all_rays[0::img_eval_interval]), total=len(train_dataset.all_rays[0::img_eval_interval]),
                            desc='calc_importance'):
        rays = samples.view(-1,samples.shape[-1])
      
        N_rays_all = rays.shape[0]
        chunk = 4096
        for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
            rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
            weight, pseudo_sampled = tensorf.forward_imp(rays_chunk, pseudo_density_planes=pseudo_density_planes, 
                                                        is_train=False, ndc_ray=args.ndc_ray)
            if weight.sum() != 0:
                (weight.detach() * pseudo_sampled).sum().backward()
                # import ipdb;ipdb.set_trace()
    importance = {}
    for idx, plane in enumerate(pseudo_density_planes):
        importance.update({f"plane_{idx}": plane.grad})
    print("Saving importance in", path)
    torch.save(importance, path)
    tensorf.importance = importance

def vector_quant(args):
    assert args.model_name == 'TensorVMSplit', 'VectorQuant only support TensorVMSplit'
    assert os.path.exists(args.ckpt), f'{args.ckpt} not exists'

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    from functools import partial
    evaluation_test = partial(evaluation,test_dataset=test_dataset, args=args, 
                        renderer=renderer,white_bg = white_bg, ndc_ray=ndc_ray,device=device,
                        compute_extra_metrics=args.autotask, im_save=args.autotask)

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
                                                        np.log(args.N_voxel_final), 
                                                        len(upsamp_list)+1))).long()).tolist()[1:]
        

    args.expname = f'{args.expname}_{args.suffix}_vq'
    logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    # dump config
    with open(os.path.join(logfolder, 'config_frozen.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    savingpath = f'{logfolder}/saving'
    os.makedirs(savingpath, exist_ok=True)

    save_name = f'{logfolder}/{args.expname}_{args.pct_high}.th'
    if os.path.exists(save_name):
        print('Load existed vq file:', save_name)
        ckpt = torch.load(save_name, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = TensorVMSplitVQ(**kwargs)
        tensorf.load(ckpt, load_vq=True)
        n_voxels = N_voxel_list[-1]
        reso_cur = N_to_reso(n_voxels, tensorf.aabb)
        nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
        tensorf.upsample_volume_grid(reso_cur)
    else:
        print('Start VQ training...')
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)

        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        kwargs.update({'codebook_size': args.codebook_size})
        kwargs.update({'use_cosine_sim': args.use_cosine_sim})
        kwargs.update({'codebook_dim': args.codebook_dim})
        tensorf = TensorVMSplitVQ(**kwargs)
        tensorf.load(ckpt, load_vq=False)

        n_voxels = N_voxel_list[-1]
        reso_cur = N_to_reso(n_voxels, tensorf.aabb)
        nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
        tensorf.upsample_volume_grid(reso_cur)

        lr_scale = 0.25
        grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
        if args.lr_decay_iters > 0:
            lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
        else:
            args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

        print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

        optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
        def reset_lr_and_optimizer():
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        torch.cuda.empty_cache()
        PSNRs,PSNRs_test = [],[0]

        allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
        if not args.ndc_ray and False:
            allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

        Ortho_reg_weight = args.Ortho_weight
        print("initial Ortho_reg_weight", Ortho_reg_weight)

        L1_reg_weight = args.L1_weight_inital
        print("initial L1_reg_weight", L1_reg_weight)
        TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
        tvreg = TVLoss()
        print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

        tensorf.saving1(savingpath)
        # tensorf.saving2(savingpath)
        # do vector quantization
        import time
        time0 = time.time()
        calc_importance(tensorf, args, logfolder, overwrite=False)
        print("==>importance cal cost:", time.time() - time0)
        if args.split_or_union == 1:
            tensorf.union_prune_and_vq(pct_mid=args.pct_mid, pct_high=args.pct_high)
        else:
            tensorf.split_prune_and_vq(pct_mid=args.pct_mid, pct_high=args.pct_high)
        
        if args.dataset_name == 'nsvf':
            tensorf.train_vq_with_mask(iteration=500, deal_reveal=2)
        elif args.dataset_name == 'llff':
            tensorf.train_vq_with_mask_imp2(importance=tensorf.importance,iteration=1000, deal_reveal=10, CHUNK=81920)
        else:
            tensorf.train_vq_with_mask(iteration=500, deal_reveal=2)

        all_indice_list = tensorf.fully_vq_both()
        reset_lr_and_optimizer()
        pbar = tqdm(range( args.vq_iters ), miniters=args.progress_refresh_rate)

        PSNRs,PSNRs_test = [0],[0]
        for iteration in pbar:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=not True)
            loss = torch.mean((rgb_map - rgb_train) ** 2)

            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {loss:.6f}'
                )
                PSNRs = []
            total_loss = loss
            if Ortho_reg_weight > 0:
                loss_reg = tensorf.vector_comp_diffs()
                total_loss += Ortho_reg_weight*loss_reg
            if L1_reg_weight > 0:
                loss_reg_L1 = tensorf.density_L1()
                total_loss += L1_reg_weight*loss_reg_L1
            if TV_weight_density>0:
                TV_weight_density *= lr_factor
                loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                total_loss = total_loss + loss_tv
            if TV_weight_app>0:
                TV_weight_app *= lr_factor
                loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
                total_loss = total_loss + loss_tv
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                PSNRs_test = evaluation_test(tensorf=tensorf, N_vis=5)

            if iteration % args.vq_up_interval == 0:
                import torch_scatter
                with torch.no_grad():
                    appplane = tensorf.app_plane # [1,c,h,w]
                    out_list_app = []
                    # import ipdb; ipdb.set_trace()
                    for i in range(len(tensorf.app_plane)):
                        data = appplane[i].reshape(tensorf.app_n_comp[i], -1).T
                        ind = all_indice_list[i].flatten()
                        out = torch_scatter.scatter(data, index=ind, dim=0, reduce='mean')
                        new_data = out[ind].T.reshape(*tensorf.app_plane[i].shape)
                        tensorf.app_plane[i].copy_(new_data)
                        out_list_app.append(out)
                    
                    denplane = tensorf.density_plane # [1,c,h,w]
                    out_list_den = []
                    for i in range(len(tensorf.density_plane)):
                        data = denplane[i].reshape(tensorf.density_n_comp[i], -1).T
                        ind = all_indice_list[i+3].flatten()
                        out = torch_scatter.scatter(data, index=ind, dim=0, reduce='mean')
                        new_data = out[ind].T.reshape(*tensorf.density_plane[i].shape)
                        tensorf.density_plane[i].copy_(new_data)
                        out_list_den.append(out)

        for i in range(len(tensorf.app_plane)):
            max_element = min(tensorf.vq[i].codebook_size, out_list_app[i].shape[0])
            tensorf.vq[i]._codebook.embed[:max_element, :] = out_list_app[i][:max_element, :]
        for i in range(len(tensorf.density_plane)):
            max_element = min(tensorf.den_vq[i].codebook_size, out_list_den[i].shape[0])
            tensorf.den_vq[i]._codebook.embed[:max_element, :] = out_list_den[i][:max_element, :]
        tensorf.fully_vq_both()
        print("==>vq training cost:", time.time() - time0)

        # save the vq-ed model
        tensorf.save(save_name)
        print("==>vq model saved in:", save_name)
    os.makedirs(f'{logfolder}/test4', exist_ok=True)
    Test4 = evaluation_test(tensorf=tensorf, N_vis=-1, savePath=f"{logfolder}/test4" if not args.debug else None, prtx=f'vq_{args.pct_high}')
    print(f'======>Test 4: AfterVQFinetune WithFinetune {args.expname} test all psnr: {np.mean(Test4)} <========================')
    tensorf.saving4(savingpath)
    

    # check the performance of the vq-ed model
    tensorf.quant()
    os.makedirs(f'{logfolder}/test5', exist_ok=True)
    Test5 = evaluation_test(tensorf=tensorf, N_vis=-1, savePath=f"{logfolder}/test5" if not args.debug else None, prtx=f'vq_quant_{args.pct_high}')
    print(f'======>Test 5: AfterVQFinetune WithFinetune(Quant) {args.expname} test all psnr: {np.mean(Test5)} <========================')

    extreme_pt = tensorf.extreme_save(logfolder)
    print('Done!')
    return extreme_pt

@torch.no_grad()
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
    TestLoad = evaluation_test(tensorf=tensorf, N_vis=-1, savePath=f"{logfolder}/extreme_load", prtx='extreme_load')
    print(f'======>Test ExtremeLoad: AfterVQFinetune WithFinetune(Quant) {args.expname} test all psnr: {np.mean(TestLoad)} <======')


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if not os.path.exists(args.ckpt):
        args.ckpt = reconstruction(args)    
    extreme_save_pt = vector_quant(args)
    args.ckpt = extreme_save_pt
    eval_vq(args)