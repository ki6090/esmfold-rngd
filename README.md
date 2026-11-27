# ESMFold on Furiosa RNGD

This web service predicts the 3D structure of a protein from its amino acid sequence. It runs Meta AI’s **ESMFold v1** using Hugging Face Transformers, two FuriosaAI RNGD accelerators, and a CPU.

Enter a single chain of **3–50 residues** using the 20 standard amino acid codes or `X` for an unknown residue. The browser displays the predicted structure colored by per-residue confidence (pLDDT) and reports the inference time. You can download the structure as a PDB file or the atomic coordinates as a CSV file.

## How inference runs

The 36 layers of the ESM-2 3B backbone run sequentially across two RNGD cards. The first card runs layers 1–18 and the second runs layers 19–36. Each card uses eight processing elements. The service pads backbone inputs to 52 tokens and masks the padding so all supported sequence lengths can use the same compilation cache.

The ESMFold folding trunk and structure module run on the CPU in FP32. The service processes one request at a time and uses three additional recycling iterations by default.

## Installation and startup

The tested configuration uses Linux, Python 3.12, two RNGD cards, PyTorch `2.10.0+cpu`, `furiosa-torch` and `furiosa-kernels` version `2026.3.0`, and Transformers `4.57.6`. Install the RNGD driver and tools on the host, then run `furiosa-smi info` to confirm that both cards are visible. Your Python environment needs access to the matching Furiosa packages and CPU PyTorch wheel. [Furiosa developer documentation](https://developer.furiosa.ai/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Install the Furiosa packages and CPU PyTorch versions listed above from your SDK package sources.
python -m pip install -r requirements.lock.txt
./download_model.sh
./service/run_service.sh start
./service/run_service.sh status
```

The download script fetches approximately 8.44 GB of model weights from revision `75a3841ee059df2bf4d56688166c8fb459ddd97a`. On first startup, the service compiles the backbone and stores the compilation cache in `cache/service/`. You can follow startup progress in `logs/service.log`.

Once the status is `ready`, open <http://127.0.0.1:8090> on the server. To connect from another computer, run the following command on that computer with your server username and hostname. Keep the SSH session open and visit the same URL in your local browser.

```bash
ssh -N -L 8090:127.0.0.1:8090 USER@HOST
```

## API example

Check service status and submit a sequence.

```bash
curl -sS http://127.0.0.1:8090/api/status
curl -sS http://127.0.0.1:8090/api/fold -H 'Content-Type: application/json' \
  -d '{"sequence":"MQIFVKTLTGKTITLEV"}'
```

Copy the `id` from the submission response and use it to check the job.

```bash
JOB_ID=replace_with_returned_id
curl -sS "http://127.0.0.1:8090/api/jobs/$JOB_ID"
```

Once the job’s `state` is `done`, download its predicted structure.

```bash
curl -f "http://127.0.0.1:8090/api/jobs/$JOB_ID/pdb" -o prediction.pdb
```

## Service commands

- Stop the service with `./service/run_service.sh stop`.
- Run the numerical compatibility tests with `.venv/bin/python -m unittest -v test_rngd_compat`.
- To try the web interface on the CPU, start the service with `./service/run_service.sh start --cpu`.

The service binds to the loopback interface by default and does not provide user authentication.

## Model credits and citation

- Lin et al. (2023), “Evolutionary-scale prediction of atomic-level protein structure with a language model,” *Science*. [Paper](https://doi.org/10.1126/science.ade2574)
- Meta AI’s FAIR research team developed ESMFold v1 and ESM-2. [Original ESM repository](https://github.com/facebookresearch/esm) · [Model card and weights](https://huggingface.co/facebook/esmfold_v1)
- This service uses the Hugging Face Transformers port of ESMFold and 3Dmol.js 2.5.3 for the browser viewer. [Transformers](https://github.com/huggingface/transformers) · [3Dmol.js](https://github.com/3dmol/3Dmol.js)

The model weights and Furiosa SDK are installed separately. License information for upstream projects and the bundled viewer is included in the repository. [Upstream notices](notices/) · [Bundled viewer notices](service/static/vendor/3Dmol-min.js.LICENSE.txt)
