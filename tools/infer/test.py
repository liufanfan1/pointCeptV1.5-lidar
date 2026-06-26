"""Pointcept 通用测试入口。

用途：
    按指定 config 和 weight 构建模型，在 config.data.test split 上推理并
    计算指标。输电线路一阶段模型、Stage-1、Stage-2 都可以用它单独测试。
输入：
    --config-file 指向训练时对应的 config.py；--options weight=... save_path=...
输出：
    save_path/test.log、save_path/result/*_pred.npy、指标汇总和运行时间 JSON。

Original author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


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
