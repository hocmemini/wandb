# YOLO + W&B + Sweeps + Registry + Launch (Starter)

End-to-end computer vision starter that **trains YOLOv8 on `coco128`**, logs to **Weights & Biases**, runs a **Sweep**,
logs the best checkpoint as a **Model Artifact**, and gives you a **Launch**-ready job spec.
Use this to demo full MLOps flow: tracking → sweeps → artifacts → (manual) registry promotion → launch jobs.
