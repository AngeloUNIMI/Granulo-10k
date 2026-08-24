# Deploying this demo to a free Hugging Face ZeroGPU Space

1. Create a new **Gradio** Space, or open the Space you already created.
2. In **Settings -> Hardware**, select **ZeroGPU**.
3. Upload all files from this directory to the root of the Space repository.
4. Upload your Experiment-1 checkpoint into `models/`.
5. Upload a small representative Granulo-10k subset into `data/Granulo-10k/`.
6. Wait for the Space to rebuild. Open **Logs** if the build or startup fails.
7. Run one prediction and confirm that the ZeroGPU allocation appears and that
   the result is multimodal rather than the image-only fallback.

For the first deployment, bundling both the model and demo subset in the Space
is the simplest configuration. Once it works, moving either asset to a separate
HF model/dataset repository is easy using the variables documented in README.md.

## Suggested demo subset

For a responsive public conference demo, start with roughly 30-100 acquisitions
covering different strand dimensions and both frontal/sideways cases. Preserve
matching A/B images, masks, point clouds, `measurements.txt`, and
`strands_ok_for_thickness.txt` entries.

## Common ZeroGPU checks

- Python is pinned to 3.12.12 in README.md.
- PyTorch is pinned to a ZeroGPU-supported 2.8.0 build.
- `spaces` is intentionally absent from requirements.txt.
- The GPU callback is decorated in `gradio_app.py`.
- The checkpoint is loaded once during Space startup.
- The bundled `pointnet2/` package removes the old dependency on the local
  training-code directory.
