import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


# def test_pipeline(root_path):
#     # parse options, set distributed setting, set ramdom seed
#     opt, _ = parse_options(root_path, is_train=False)

#     torch.backends.cudnn.benchmark = True
#     # torch.backends.cudnn.deterministic = True

#     # mkdir and initialize loggers
#     make_exp_dirs(opt)
#     log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
#     logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
#     logger.info(get_env_info())
#     logger.info(dict2str(opt))

#     # create test dataset and dataloader
#     test_loaders = []
#     for _, dataset_opt in sorted(opt['datasets'].items()):
#         test_set = build_dataset(dataset_opt)
#         test_loader = build_dataloader(
#             test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#         logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
#         test_loaders.append(test_loader)

#     # create model
#     model = build_model(opt)

#     for test_loader in test_loaders:
#         test_set_name = test_loader.dataset.opt['name'] 
#         logger.info(f'Testing {test_set_name}...')
#         print('save_img:',opt['val']['save_img'])
#         model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # test_pipeline(root_path)
    print('root_path:',root_path)
    import torchvision.models as models
    import torch
    from ptflops import get_model_complexity_info

    opt, _ = parse_options(root_path, is_train=False)
    with torch.cuda.device(0):
    #   net = models.densenet161()
        # create model
        model = build_model(opt)
        model = model.net_g 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!:",type(model))
        macs, params = get_model_complexity_info(model, (3, 360, 640), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))