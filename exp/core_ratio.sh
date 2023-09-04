for i in {0.99,0.95,0.90,0.80}
do
python main.py --config-name=cifar10_resnet20 script=prune name=cifar10-resnet20/ablation_$1_$2/$i \
        model.name=resnet20 data.params.batch_size=128 optimizer.params.lr=0.1 \
        prune.amount=${i} prune\/method=pai prune\/score=$1 prune.method.iterations=$3 \
        model.load_path=$PWD/logs/cifar10-resnet20/bl/saved model.load_name=best prune.pwr=False \
        +prune.pai_ratio=True +prune.pai_new_iterations=$4 +prune.pai_score=$2 prune.method.finetune=160

python main.py --config-name=tinyimagenet_resnet18 script=prune name=tinyimagenet-resnet18_modified/ablation_$1_$2/$i \
        model.name=resnet18_modified data.params.batch_size=256 optimizer.params.lr=0.1 \
        prune.amount=${i} prune\/method=pai prune\/score=$1 prune.method.iterations=$3 \
        model.load_path=$PWD/logs/tinyimagenet-resnet18_modified/bl/saved model.load_name=best prune.pwr=False \
        +prune.pai_ratio=True +prune.pai_new_iterations=$4 +prune.pai_score=$2 prune.method.finetune=160
done
