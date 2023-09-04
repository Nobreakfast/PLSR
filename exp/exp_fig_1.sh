for i in {"conv1","layer1.0.conv1","layer1.0.conv2","layer2.0.conv1","layer2.0.conv2","layer3.0.conv1","layer3.0.conv2","fc"}
do
python main.py --config-name=cifar10_resnet8 name=train-resnet8-cifar10-trained/${i} model.params.pretrained=$i epoch=160 \
    model.pretrained=True model.load_path=$PWD/logs/train-resnet8-cifar10/no/saved model.load_name=best
done
