for i in {0.99,0.95,0.90,0.80}
do
    python main.py --config-name=cifar10_resnet20 script=prune \
            model.name=resnet20 data.params.batch_size=128 optimizer.params.lr=0.1 \
            prune.amount=${i} prune\/method=$1 prune\/score=$2 prune.method.iterations=$3 \
            model.load_path=$PWD/logs/cifar10-resnet20/bl/saved \
            model.load_name=best prune.pwr=$4 prune.method.finetune=160 +prune.pai_ratio=False

    python main.py --config-name=tinyimagenet_resnet18 script=prune \
            model.name=resnet18_modified data.params.batch_size=256 optimizer.params.lr=0.1 \
            prune.amount=${i} prune\/method=$1 prune\/score=$2 prune.method.iterations=$3 \
            prune.load_path=$PWD/logs/tinyimagenet-resnet18_modified/bl/saved \
            prune.load_name=best prune.pwr=$4 prune.method.finetune=160 +prune.pai_ratio=False
done
