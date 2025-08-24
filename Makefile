help:
	@echo "make train | sweep | eval"
train:
	python train.py --project yolo-demo --epochs 5 --imgsz 640
sweep:
	wandb sweep sweep.yaml
eval:
	python eval_log.py --project yolo-demo --imgsz 640
