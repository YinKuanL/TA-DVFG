# Recorded Experimental Environment

`system_info.json` and `package_versions.txt` are transcriptions of the existing
environment evidence in `outputs/reproducibility/environment/`. They describe the
reported experiment machine; they are not a fresh snapshot of the machine running
this cleanup. No hardware or version value was inferred.

The host contained an NVIDIA GPU and CUDA 12.6 toolkit, but the recorded PyTorch
build was `2.12.1+cpu`, reported `cuda_available=False`, and ran the experiments
on CPU. `requirements.txt` is the minimal dependency declaration; the version
snapshot is provenance, not a promise that every unrelated package is required.

