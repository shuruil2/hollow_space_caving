import numpy as np
import torch
import argparse
import random
from model.utils import set_gpu
from model.trainer.fsl_trainer import FSLTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=600)

    # 模型与骨干
    parser.add_argument('--model_class', type=str, default='HSC',
                        choices=['BILSTM', 'DeepSet', 'GCN', 'FEAT', 'HSC', 'HSCModel'])
    parser.add_argument('--use_euclidean', action='store_true', help='(FEAT专用) 使用欧氏距离分类')
    parser.add_argument('--backbone_class', type=str, default='Res18',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])

    # 数据集
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'Cifar100', 'Caltech'])

    # episodic 配置（闭集/开集 way、shot、query）
    parser.add_argument('--closed_way', type=int, default=5)
    parser.add_argument('--open_way', type=int, default=5)               # 训练中的开集 way（若用到）
    parser.add_argument('--closed_eval_way', type=int, default=5)
    parser.add_argument('--open_eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)

    # 原项目参数（保持兼容）
    parser.add_argument('--balance', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=64)
    parser.add_argument('--temperature2', type=float, default=64)

    # -------------------------
    # 优化器与调度器
    # -------------------------
    parser.add_argument('--orig_imsize', type=int, default=-1)  # -1: no cache, -2: no resize
    parser.add_argument('--lr', type=float, default=2e-4)       # backbone/base lr
    parser.add_argument('--head_lr', type=float, default=1e-3)  # 头部更大学习率（HSC推荐）
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--fix_BN', action='store_true', default=False)   # 只是不更新 BN 的 running 统计
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str,
                        default='./saves/initialization/miniimagenet/Res18-pre.pth')

    # 通常不改
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_path', type=str, default=None)

    # EDL/开集（保留兼容）
    parser.add_argument('--annealing_step', type=float, default=10)
    parser.add_argument('--loss_type_no', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='edl_loss', choices=['edl_loss', 'ce_loss'])
    parser.add_argument('--edl_loss', type=str, default='mse')
    parser.add_argument('--open_loss', action='store_true', help='(原FEAT开集项)')

    parser.add_argument('--open_loss_coeff', type=float, default=1.0)

    # -------------------------
    # HSC（雕刻空域）专用超参数
    # -------------------------
    parser.add_argument('--kappa', type=float, default=10.0, help='超球余弦缩放')
    parser.add_argument('--open_margin', type=float, default=0.1, help='已知类 margin（空出边界）')
    parser.add_argument('--lambda_energy', type=float, default=1.0, help='能量整形损失权重')
    parser.add_argument('--lambda_kl', type=float, default=1.0, help='未知证据KL损失权重')
    parser.add_argument('--lambda_edl', type=float, default=1.0, help='(可选) 闭集EDL损失权重')
    parser.add_argument('--diffusion_steps', type=int, default=3, help='能量扩散步数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='energy/evidence head 的隐层维度')

    # 能量目标（已知低、未知高）与扩散步长
    parser.add_argument('--e_target_closed', type=float, default=0.1)
    parser.add_argument('--e_target_unknown', type=float, default=0.9)
    parser.add_argument('--diff_step_size', type=float, default=0.1)

    args = parser.parse_args()

    # 兼容旧选项：loss_type_no 控制 loss_type
    if args.loss_type_no == 0:
        args.loss_type = 'edl_loss'
    else:
        args.loss_type = 'ce_loss'

    # 评估 episodic 配置
    args.eval_shot = args.shot
    args.eval_way = args.closed_way + args.open_eval_way

    # 数据路径与大小写修正
    if args.dataset == 'MiniImageNet':
        args.data_path = args.data_path or './data/miniimagenet'
    elif args.dataset == 'TieredImageNet':
        args.data_path = args.data_path or './data/tieredimagenet'
    elif args.dataset == 'Cifar100':
        args.data_path = args.data_path or './data/cifar100'
    elif args.dataset == 'Caltech':
        args.data_path = args.data_path or './data/caltech'

    if args.data_path is None:
        raise ValueError('Specify your data path')

    # 保存路径/权重名
    args.save_path = args.save_dir
    prefix = ('mini' if args.dataset == 'MiniImageNet'
              else 'tiered' if args.dataset == 'TieredImageNet'
              else 'cifar' if args.dataset == 'Cifar100'
              else 'caltech')
    args.weight_name = '%s-%s-%d-shot.pth' % (prefix, args.model_class.lower(), args.shot)

    # 随机性控制
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # GPU
    set_gpu(args.gpu)

    # 运行
    trainer = FSLTrainer(args)
    trainer.train()
    trainer.evaluate_test()
    trainer.final_record()
    print(args.save_path)


