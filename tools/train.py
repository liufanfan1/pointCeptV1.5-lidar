"""Pointcept 通用训练入口。

用途：
    按指定 config 训练模型。输电线路一阶段模型、两阶段中的 Stage-1 和
    Stage-2 都是通过这个入口启动训练，只是 config 和数据集不同。
输入：
    --config-file 指向 configs/transmission_line 或其他数据集配置；
    --options 可覆盖 save_path、resume、weight 等运行参数。
输出：
    exp/.../model/model_last.pth、model_best.pth、train.log、config.py。

Original author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
