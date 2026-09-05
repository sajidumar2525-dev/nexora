# Nexora — free cloud deployment

## What changed

The Windows-only `realesrgan-ncnn-vulkan.exe` backend was replaced with the Python Real-ESRGAN backend, so the app can run on Linux hosting such as Streamlit Community Cloud.

The existing `RealESRGAN_x4plus.pth` model is kept in `weights/`.

The 2x and 3x options use the same x4 model with Real-ESRGAN's arbitrary `outscale` support.

## Folder structure

```text
Nexora/
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
└── weights/
    └── RealESRGAN_x4plus.pth
```

## Deploy

1. Create a GitHub repository, e.g. `nexora`.
2. Upload these files/folders.
3. Open Streamlit Community Cloud.
4. Choose **Deploy an app**.
5. Select the GitHub repository and `app.py`.
6. Use Python 3.12 (or another supported Python version).
7. Deploy.

The resulting public URL will be a `*.streamlit.app` address.

## Important

This free deployment is CPU-only. Real-ESRGAN will be considerably slower than the Vulkan GPU executable running on your PC. Very large images can also exceed the free instance's memory; the app therefore protects the server by limiting estimated output size.

Do not upload the Windows `realesrgan-ncnn-vulkan.exe` to the cloud deployment.
