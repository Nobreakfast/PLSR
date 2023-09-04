case $1 in
"train")
  python main.py script=train task.epoch=135
  ;;
"eval")
  python main.py script=eval \
    model.name=resnet18_new model.load_path=$2 model.load_name=$3 \
    model.pt=True
  ;;
"finetune")
  python main.py script=train \
    model.name=resnet18_new model.load_path=$2 model.load_name=$3 \
    model.pt=True
  ;;
"ratio")
  python main.py name=ratio script=prune prune\/method=ratio \
  prune\/score=rand prune.amount=0.99
  ;;
"ratio2")
  python main.py name=ratio2 script=prune prune\/method=ratio \
  prune\/score=rand prune.amount=0.99 prune.method.name=ratio2
  ;;
"pai_random")
  python main.py name=pai_random script=prune prune\/method=pai \
    prune\/score=rand_amount prune.amount=0.99 \
    prune.method.iterations=100 prune.method.finetune=0 \
    prune.random=False
  ;;
"pai_random_res")
  python main.py name=pai_random_res script=prune prune\/method=pai \
    prune\/score=rand_amount prune.amount=0.99 \
    prune.method.iterations=100 prune.method.finetune=0 \
    prune.random=False prune.restore=True
  ;;
"pai_random_restore")
  python main.py name=pai_random_restore script=prune prune\/method=pai \
    prune.method.name=pai_restore \
    prune\/score=rand prune.amount=0.99 prune.score.name=rand_amount \
    prune.method.iterations=100 prune.method.finetune=0
  ;;
"imp")
  python main.py name=imp script=prune prune.method.epoch=1 prune.amount=0.99
  ;;
"lth")
  python main.py name=lth script=prune prune\/method=lth prune.amount=0.99
  ;;
"ltr")
  python main.py name=ltr script=prune prune\/method=ltr prune.amount=0.99
  ;;
"synflow")
  # shellcheck disable=SC1101
  python main.py -m name=synflow script=prune \
    prune\/method=pai \
    prune\/score=synflow \
    prune.amount=0.99 \
    prune.method.finetune=0
  ;;
"synflow_res")
  # shellcheck disable=SC1101
  python main.py -m name=synflow_res script=prune \
    prune\/method=pai \
    prune\/score=synflow \
    prune.amount=0.99 \
    prune.method.finetune=0 prune.restore=True
  ;;
"snip")
  # shellcheck disable=SC1101
  python main.py -m name=snip script=prune \
    prune\/method=pai \
    prune.score.name=snip\
    prune.amount=0.99 \
    prune.method.finetune=0
  ;;
"sepeva")
  # shellcheck disable=SC1101
  python main.py -m name=sepeva script=prune \
    prune\/method=pai \
    prune\/score=sepeva \
    prune.amount=0.99 \
    prune.method.finetune=0
  ;;
"sepeva_lambda")
  # shellcheck disable=SC1101
  python main.py -m name=sepeva_lambda script=prune \
    prune\/method=pai \
    prune\/score=sepeva \
    prune.score.name=sepeva_lambda \
    prune.amount=0.99 \
    prune.method.finetune=0
  ;;
"check_model")
  python main.py script=check_model model.name=$2
  ;;
"check")
  python PWR/check.py
  ;;
"open_log")
  tensorboard --logdir=logs
  ;;
*)
  echo "no argv detected"
  ;;
esac
