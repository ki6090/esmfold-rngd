"""ESMFold v1 inference engine for the web service.

Same device split as esmfold_rngd.py: the ESM-2 3B backbone (36 layers) runs on
RNGD as compiled blocks; the folding trunk, structure module and heads run on
CPU FP32.

RNGD blocks are compiled for one fixed shape, so every request is right-padded
to BUCKET_TOKENS (BOS + MAX_LEN + EOS) and padded keys are masked out. ESM-2
uses rotary (relative) positions and zeroes padded embeddings, so real-token
outputs match the unpadded model and a single compile serves every length.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import time
import types

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import esmfold_rngd as base  # noqa: E402

MAX_LEN = 50
MIN_LEN = 3
BUCKET_TOKENS = MAX_LEN + 2
RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX")
TRUNK_BLOCKS = 48


def clean_sequence(text: str) -> str:
    """Accept raw or FASTA text; return an upper-case sequence or raise ValueError."""
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]
    seq = "".join("".join(lines).split()).upper()
    if not seq:
        raise ValueError("서열이 비어 있습니다.")
    invalid = sorted(set(seq) - RESIDUES)
    if invalid:
        raise ValueError(f"허용되지 않는 문자: {' '.join(invalid)} (아미노산 1글자 코드 20종 + X만 가능)")
    if not MIN_LEN <= len(seq) <= MAX_LEN:
        raise ValueError(f"길이 {len(seq)} aa: {MIN_LEN}~{MAX_LEN} aa만 지원합니다.")
    return seq


class Engine:
    def __init__(self, *, use_npu=True, devices=("furiosa:0", "furiosa:1"), recycles=3, threads=12,
                 chunk_size=64, cache_dir=ROOT / "cache/service", model_dir=ROOT / "models/esmfold_v1"):
        self.use_npu, self.devices, self.recycles, self.threads = use_npu, list(devices), recycles, threads
        self.cache_dir = Path(cache_dir)
        self.backend = self.compiler_config = None
        if use_npu:
            import furiosa.torch as ft
            ft.set_fusion(8)  # Must precede any device allocation/query.
            config = ft.config.CompilerConfig()  # Same settings as esmfold_rngd.py (cache-compatible).
            config.max_num_unique_shapes = 8
            config.local_population_threshold = 2
            config.expected_total_beam_states = 2000
            config.tactic_sorting_policy = ft.config.TacticSortingPolicy.ByEstimation
            self.compiler_config = config
            self.backend = ft.backend.with_config(compiler_config=config, eager_fallback=False)
        torch.set_num_threads(threads)
        self.model = EsmForProteinFolding.from_pretrained(
            model_dir, local_files_only=True, low_cpu_mem_usage=True).float().eval()
        self.model.trunk.set_chunk_size(chunk_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.parameters_b = sum(p.numel() for p in self.model.parameters()) / 1e9
        self.npu_ready = False
        self.npu_resident: dict = {}
        self.progress: dict = {"stage": "idle"}
        self.timings: dict = {}
        self._marks: dict = {}
        self._stack = ExitStack()
        self._install_hooks()

    # -- setup -------------------------------------------------------------------------------
    def enter_context(self):
        """Activate the RNGD compile cache on the calling thread (the thread that runs folds)."""
        if self.use_npu:
            import furiosa.torch as ft
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._stack.enter_context(
                ft.config.context(compiler_config=self.compiler_config, cache_dir=str(self.cache_dir)))

    def install_padded_lm(self):
        self.model.compute_language_model_representations = types.MethodType(self._padded_lm(), self.model)

    def prepare_npu(self):
        """Move the 36 backbone layers onto RNGD (compiled lazily on first call)."""
        if self.use_npu and not self.npu_ready:
            base.prepare(self.model, 36, 0, self.devices, self.backend)
            self.npu_ready = True
            # Per-chip residency as placed by us (furiosa-smi does not count furiosa-torch allocations).
            for mod in self.model.modules():
                if isinstance(mod, base.NPUBlock):
                    entry = self.npu_resident.setdefault(str(mod.device), {"bytes": 0, "layers": []})
                    entry["bytes"] += sum(p.numel() * p.element_size() for p in mod.parameters())
                    entry["layers"].append(int(mod.label.split(".")[1]))
        self.install_padded_lm()  # base.prepare installs the unpadded variant.

    def _padded_lm(engine):
        def representations(self, esmaa):
            start = time.monotonic()
            b, length = esmaa.shape
            if length > MAX_LEN:
                raise ValueError(f"length {length} exceeds bucket {MAX_LEN}")
            tokens = esmaa.new_full((b, BUCKET_TOKENS), self.esm_dict_padding_idx)
            tokens[:, 0] = self.esm_dict_cls_idx
            tokens[:, 1:length + 1] = esmaa
            tokens[:, length + 1] = self.esm_dict_eos_idx
            valid = tokens != self.esm_dict_padding_idx
            h = self.esm.embeddings(input_ids=tokens, attention_mask=valid.long())
            # Finite (not -inf/finfo.min) so the additive mask stays representable in BF16 on RNGD.
            mask = (~valid).float()[:, None, None, :] * -1e9
            states = [h]
            layers = self.esm.encoder.layer
            for i, layer in enumerate(layers):
                engine.progress.update(stage="backbone", backbone=i)
                h = layer(h, attention_mask=mask)
                states.append(h.cpu())
            engine.progress.update(backbone=len(layers))
            states[-1] = self.esm.encoder.emb_layer_norm_after(h.cpu())
            engine.timings["language_model"] = time.monotonic() - start
            return torch.stack(states, dim=2)[:, 1:length + 1]
        return representations

    def _install_hooks(self):
        # The trunk runs recycles+1 passes and calls the structure module once per pass
        # (its coordinates feed the next recycle), so structure time is accumulated.
        trunk, marks, progress = self.model.trunk, self._marks, lambda: self.progress

        def trunk_start(*_):
            marks["trunk_start"] = time.monotonic()

        def trunk_end(*_):
            marks["trunk_end"] = time.monotonic()
            progress()["stage"] = "heads"

        def block_done(*_):
            p = progress()
            p["trunk"] = p.get("trunk", 0) + 1
            p["stage"] = "trunk"

        def structure_start(*_):
            marks["structure_start"] = time.monotonic()
            p = progress()
            p["structure"] = p.get("structure", 0) + 1
            p["stage"] = "structure"

        def structure_end(*_):
            marks["structure"] = marks.get("structure", 0.0) + time.monotonic() - marks["structure_start"]

        trunk.register_forward_pre_hook(trunk_start)
        trunk.register_forward_hook(trunk_end)
        trunk.structure_module.register_forward_pre_hook(structure_start)
        trunk.structure_module.register_forward_hook(structure_end)
        for block in trunk.blocks:
            block.register_forward_hook(block_done)

    # -- inference ---------------------------------------------------------------------------
    def new_progress(self):
        passes = self.recycles + 1
        return {"stage": "tokenize", "backbone": 0, "backbone_total": 36, "trunk": 0,
                "trunk_total": TRUNK_BLOCKS * passes, "structure": 0, "passes": passes}

    def run_model(self, seq):
        self.timings = {}
        self._marks.clear()  # Hooks hold a reference to this dict.
        self.progress = self.new_progress()
        inputs = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        start = time.monotonic()
        with torch.no_grad():
            out = self.model(**inputs, num_recycles=self.recycles)
        return out, time.monotonic() - start

    def fold(self, seq):
        seq = clean_sequence(seq)
        npu_blocks = [mod for mod in self.model.modules() if isinstance(mod, base.NPUBlock)]
        calls_before = sum(mod.calls for mod in npu_blocks)
        out, total = self.run_model(seq)
        self.progress["stage"] = "postprocess"
        info, ca = base.summarize(out, seq, total)
        pdb = base.pdb_string(self.model, out)
        m = self._marks
        trunk_total = m.get("trunk_end", 0) - m.get("trunk_start", 0)
        structure = m.get("structure", 0.0)
        language_model = self.timings.get("language_model", 0.0)
        placement = {}
        for mod in npu_blocks:
            placement.setdefault(str(mod.device), []).append(int(mod.label.split(".")[1]))
        self.progress["stage"] = "done"
        return {
            "sequence": seq,
            "length": len(seq),
            "mean_plddt": info["mean_plddt"],
            "ptm": info["ptm"],
            "plddt": (out.plddt[0, :, 1] * 100).tolist(),  # per-residue (CA), 0-100
            "ca": ca.tolist(),
            "pdb": pdb,
            "recycles": self.recycles,
            "mode": "rngd-hybrid" if npu_blocks else "cpu",
            "backbone_layers_on_npu": len(npu_blocks),
            "npu_placement": {dev: f"layers {min(ix)}-{max(ix)}" for dev, ix in placement.items()},
            "npu_block_calls": sum(mod.calls for mod in npu_blocks) - calls_before,  # this request only
            "bucket_tokens": BUCKET_TOKENS,
            "timings": {
                "total": total,
                "language_model": language_model,
                "folding_trunk": max(trunk_total - structure, 0.0),
                "structure_module": structure,
                "other": max(total - language_model - trunk_total, 0.0),
            },
        }
