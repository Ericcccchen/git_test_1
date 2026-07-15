# EmotionAI

A browser-based facial emotion recognition demo using a trained ResNet-34 model.

The GitHub Pages version runs inference locally in the browser with ONNX Runtime Web. Uploaded images are not sent to a server.

## Recognized emotions

Disgust, fear, happy, neutral, sad, and surprise.

## Project structure

- `index.html` — project landing page
- `tool.html` — browser inference interface
- `models/emotion.onnx` — exported inference model
- `source/` — original Flask and training source files
- `scripts/export_onnx.py` — reproducible model export script

## Local preview

```bash
python3 -m http.server 8000
```

Inference runs entirely in the visitor's browser. Images are not uploaded or stored.
