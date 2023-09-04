# check the logs path in logs/*.txt file, find the pretrained models

# train a baseline model
python main.py name=cifar10_resnet20/bl script=train \
              model=resnet model.name=resnet20 data=cifar10 \
              optimizer=sgd optimizer.params.lr=0.1 \
              scheduler=cyc_tri2

python main.py name=tinyimagenet_resnet18/bl script=train \
               model=resnet model.name=resnet18_modified \
               data=tinyimagenet data.params.batch_size=256 \
               optimizer=sgd optimizer.params.lr=0.2 \
               scheduler=cyc_tri2