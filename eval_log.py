
import argparse, glob, pathlib, numpy as np, wandb
from ultralytics import YOLO
def guess_coco128_val_dirs():
    candidates = []
    home = pathlib.Path.home()
    common = [pathlib.Path("datasets/coco128/images/val2017"),
              pathlib.Path("ultralytics/datasets/coco128/images/val2017"),
              home / ".cache" / "ultralytics" / "coco128" / "images" / "val2017",
              home / "datasets" / "coco128" / "images" / "val2017"]
    for p in common:
        if p.exists(): candidates.append(p)
    return candidates
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--project", type=str, default="yolo-demo")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ckpt", type=str, default="runs/train/wb_yolo_demo/weights/best.pt")
    args = ap.parse_args()
    run = wandb.init(project=args.project, entity=args.entity, job_type="evaluation")
    model = YOLO(args.ckpt)
    res = model.val(data="coco128.yaml", imgsz=args.imgsz, verbose=True)
    try:
        metrics = {"metrics/precision": float(getattr(res.box, "mp", np.nan)),
                   "metrics/recall": float(getattr(res.box, "mr", np.nan)),
                   "metrics/mAP50": float(getattr(res.box, "map50", np.nan)),
                   "metrics/mAP50-95": float(getattr(res.box, "map", np.nan))}
        wandb.log(metrics)
    except Exception as e:
        print("Metric logging note:", e)
    val_dirs = guess_coco128_val_dirs()
    if val_dirs:
        val_dir = val_dirs[0]
        imgs = sorted(glob.glob(str(val_dir / "*.jpg")))[:24]
        if imgs:
            preds = model.predict(source=imgs, imgsz=args.imgsz, conf=0.25, save=False, verbose=False)
            media = []
            for img_path, pred in zip(imgs, preds):
                boxes = getattr(pred, "boxes", None)
                if boxes is None or boxes.xyxy is None: continue
                arr = boxes.xyxy.cpu().numpy()
                cls = boxes.cls.cpu().numpy() if boxes.cls is not None else None
                scores = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                box_data = []
                for i, (x1, y1, x2, y2) in enumerate(arr):
                    c = int(cls[i]) if cls is not None else 0
                    s = float(scores[i]) if scores is not None else 0.0
                    box_data.append({"position": {"minX": float(x1), "minY": float(y1), "maxX": float(x2), "maxY": float(y2)},
                                     "class_id": c, "box_caption": f"class={c} conf={s:.2f}"})
                classes = {int(i): f"class_{int(i)}" for i in set(cls.tolist())} if cls is not None else {0: "obj"}
                media.append(wandb.Image(img_path, boxes={"predictions": {"box_data": box_data, "class_labels": classes}}))
            if media: wandb.log({"predictions": media})
    else:
        print("Could not locate coco128 val images; logged metrics only.")
    run.finish()
if __name__ == "__main__":
    main()
