# Release Checklist

## Before the first public release

- [x] Add the MIT application license.
- [ ] Replace any private names or paths in documentation.
- [ ] Confirm `settings.json`, `.venv`, logs, input/output images, and `.pt` models are ignored.
- [ ] Confirm no copyrighted or private test images are included.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Test one image on CPU-only mode.
- [ ] Test one image on an NVIDIA GPU when available.
- [ ] Test model download using `DOWNLOAD_RECOMMENDED_MODEL.bat`.
- [ ] Build the Windows application and launch it on a clean environment.
- [ ] Confirm the output still requires human review and state this in the release notes.

## GitHub Release

- [ ] Create a version tag such as `v1.0.0`.
- [ ] Attach the application as a ZIP if distributing a built version.
- [ ] Do not include `nsfw-anime-xl-x1280.pt` in the repository or release ZIP unless its redistribution terms have been rechecked.
- [ ] Link to https://huggingface.co/01miku/anime-nsfw-segm-yolo26.
- [ ] Mention that CPU works but is slower, while NVIDIA GPU is used automatically.
