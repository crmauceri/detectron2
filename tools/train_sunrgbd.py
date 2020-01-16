from .train_net import main
from detectron2.engine import launch, default_argument_parser
from detectron2.data.datasets import register_coco_instances

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    for d in ["train", "val"]:
        register_coco_instances("sunrgbd_{}".format(d), {},
                                "/Users/Mauceri/Workspace/SUNRGBD/annotations/instances_{}.json".format(d),
                                "/Users/Mauceri/Workspace/", "/Users/Mauceri/Workspace/")

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )