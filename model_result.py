from ultralytics import YOLO

if __name__ == "__main__":
    # โหลดโมเดล
    model = YOLO(r"/runs/detect/train/weights/best.pt")
    data = r"your_dataset_path/data.yaml"

    # Train split
    metrics_train = model.val(split="train", data=data)
    print("📊 Train metrics:", metrics_train.results_dict)

    # Validation split
    metrics_val = model.val(split="val", data=data)
    print("📊 Val metrics:", metrics_val.results_dict)

    # Test split (ถ้ามี)
    try:
        metrics_test = model.val(split="test", data=data)
        print("📊 Test metrics:", metrics_test.results_dict)
    except Exception as e:
        print("⚠️ ไม่มี test set หรือเกิด error:", e)
