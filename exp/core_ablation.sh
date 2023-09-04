for i in {0.99,0.95,0.90,0.80}
do
    python main.py --config-name=cifar10_resnet20 name=cifar10-resnet20/ablation_$1_$2_$3 script=prune \
            model.name=resnet20 data.params.batch_size=128 optimizer.params.lr=0.1 \
            prune.amount=${i} prune\/method=pai prune\/score=featio prune.method.iterations=50 \
            model.load_path=$PWD/logs/cifar10-resnet20/bl/saved \
            model.load_name=best prune.pwr=False prune.method.finetune=160 +prune.pai_ratio=False \
            prune.shuffle=$1 prune.prune_init=$2 prune.reinit=$3

    python main.py --config-name=tinyimagenet_resnet18 name=tinyimagenet-resnet18_modified/ablation_$1_$2_$3 script=prune \
            model.name=resnet18_modified data.params.batch_size=256 optimizer.params.lr=0.1 \
            prune.amount=${i} prune\/method=pai prune\/score=featio prune.method.iterations=50 \
            prune.load_path=$PWD/logs/tinyimagenet-resnet18_modified/bl/saved \
            prune.load_name=best prune.pwr=False prune.method.finetune=160 +prune.pai_ratio=False \
            prune.shuffle=$1 prune.prune_init=$2 prune.reinit=$3
done
