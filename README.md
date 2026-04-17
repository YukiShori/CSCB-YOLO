
## 📄 Documentation

See below for quickstart installation and usage examples. For comprehensive guidance on training, validation, prediction, and deployment, refer to  full [Ultralytics Docs](https://docs.ultralytics.com/).




<details open>
<summary>Usage</summary>

##  CSCB


### Python

Ultralytics YOLO can also be integrated directly into your Python projects. It accepts the same [configuration arguments](https://docs.ultralytics.com/usage/cfg/) as the CLI:

```python
from ultralytics import YOLO

def main():

# Train the model
    model = YOLO(model = "ultralytics/cfg/models/v8/CSCB/adev2.yaml")
    results = model.train(data="data.yaml", epochs=600, imgsz=640 , batch=16, optimizer='AdamW',mosaic=0.3, close_mosaic=15, lr0=0.005)

if __name__ == '__main__':
    main()
```

</details>
