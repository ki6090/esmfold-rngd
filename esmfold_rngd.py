"""ESMFold inference on RNGD, with explicit CPU geometry and strict NPU blocks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
import types

import torch
from torch import nn
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm import modeling_esm as esm

ROOT = Path(__file__).resolve().parent
START = time.monotonic()


def log(message):
    print(f"[{time.monotonic() - START:8.1f}s] {message}", flush=True)


def rotary_forward(self, q, k):
    t = torch.arange(k.shape[-2], device=k.device, dtype=torch.float32)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    return esm.apply_rotary_pos_emb(q, cos, sin), esm.apply_rotary_pos_emb(k, cos, sin)


def attention_forward(self, hidden_states, attention_mask=None, head_mask=None, **kwargs):
    b, length, _ = hidden_states.shape
    shape = (b, length, self.num_attention_heads, self.attention_head_size)
    q = self.query(hidden_states).view(shape).transpose(1, 2)
    k = self.key(hidden_states).view(shape).transpose(1, 2)
    v = self.value(hidden_states).view(shape).transpose(1, 2)
    q = q * self.attention_head_size**-0.5
    q, k = self.rotary_embeddings(q, k)
    scores = q @ k.transpose(-1, -2)
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = scores.softmax(-1)
    if head_mask is not None:
        probs = probs * head_mask
    return (probs @ v).transpose(1, 2).reshape(b, length, -1), probs


def fold_attention_forward(self, x, mask=None, bias=None, indices=None):
    t = self.proj(x).view(*x.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
    q, k, v = t.chunk(3, dim=-1)
    a = (q * self.rescale_factor) @ k.transpose(-1, -2)
    if bias is not None:
        a = a + bias.permute(0, 3, 1, 2)
    if mask is not None:
        a = a.masked_fill(mask[:, None, None] == 0, float("-inf"))
    a = a.softmax(-1)
    y = (a @ v).permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)
    if self.gated:
        y = self.g_proj(x).sigmoid() * y
    return self.o_proj(y), a.permute(0, 3, 1, 2)


def language_representations(self, esmaa):
    """Collect states explicitly: HF forward-capture hooks cannot straddle devices."""
    b, _ = esmaa.shape
    bos = esmaa.new_full((b, 1), self.esm_dict_cls_idx)
    pad = esmaa.new_full((b, 1), self.esm_dict_padding_idx)
    tokens = torch.cat((bos, esmaa, pad), dim=1)
    tokens[torch.arange(b), (tokens != self.esm_dict_padding_idx).sum(1)] = self.esm_dict_eos_idx
    valid = tokens != self.esm_dict_padding_idx
    # Matches EsmModel.forward in pinned transformers 4.57.6.
    h = self.esm.embeddings(input_ids=tokens)
    mask = self.esm.get_extended_attention_mask(valid, tokens.shape)
    states = [h]
    for layer in self.esm.encoder.layer:
        h = layer(h, attention_mask=mask)
        states.append(h.cpu())
    states[-1] = self.esm.encoder.emb_layer_norm_after(h.cpu())
    return torch.stack(states, dim=2)[:, 1:-1]


class NPUBlock(nn.Module):
    def __init__(self, inner, device, name, backend):
        super().__init__()
        self.device, self.label, self.calls = torch.device(device), name, 0
        self.compiled = torch.compile(inner.to(self.device), backend=backend, fullgraph=True, dynamic=False)

    def forward(self, *args, **kwargs):
        from torch.utils._pytree import tree_map
        to_device = lambda t: t.to(self.device) if isinstance(t, torch.Tensor) else t
        to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
        self.calls += 1
        if self.calls == 1:
            log(f"{self.label} -> {self.device}: first call / compile")
        result = self.compiled(*tree_map(to_device, args), **tree_map(to_device, kwargs))
        result = tree_map(to_cpu, result)
        if self.calls == 1:
            log(f"{self.label}: NPU completed")
        return result


def prepare(model, backbone_layers, trunk_blocks, devices, backend):
    model.compute_language_model_representations = types.MethodType(language_representations, model)
    for i, layer in enumerate(model.esm.encoder.layer):
        attn = layer.attention.self
        attn.forward = types.MethodType(attention_forward, attn)
        attn.rotary_embeddings.forward = types.MethodType(rotary_forward, attn.rotary_embeddings)
        if i < backbone_layers:
            dev = devices[min(i * len(devices) // backbone_layers, len(devices) - 1)]
            model.esm.encoder.layer[i] = NPUBlock(layer, dev, f"backbone.{i}", backend)
    for i in range(trunk_blocks):
        block = model.trunk.blocks[i]
        block.seq_attention.forward = types.MethodType(fold_attention_forward, block.seq_attention)
        dev = devices[min(i * len(devices) // trunk_blocks, len(devices) - 1)]
        model.trunk.blocks[i] = NPUBlock(block, dev, f"trunk.{i}", backend)


def read_sequences(args):
    if args.sequence:
        records = [("sequence", args.sequence)]
    else:
        records = []
        name, chunks = None, []
        for line in args.fasta.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].strip(), []
            elif name is None:
                raise ValueError("FASTA must start with a >header")
            else:
                chunks.append(line)
        if name is not None:
            records.append((name, "".join(chunks)))
    if not records:
        raise ValueError("No sequences found")
    cleaned = []
    for name, seq in records:
        seq = "".join(seq.split()).upper()
        invalid = set(seq) - set("ARNDCQEGHILKMFPSTWYVX")
        if not seq or invalid:
            raise ValueError(f"Invalid protein sequence {name!r}: {sorted(invalid)}")
        if len(seq) > args.max_length:
            raise ValueError(f"{name}: length {len(seq)} exceeds --max-length {args.max_length}; no truncation performed")
        cleaned.append((name, seq))
    return cleaned


def summarize(out, seq, elapsed):
    ca = out.positions[-1, 0, :, 1].float().cpu()
    if not torch.isfinite(ca).all() or not torch.isfinite(out.plddt).all():
        raise RuntimeError("Non-finite model output")
    atom_mask = out.atom37_atom_exists
    return {
        "sequence": seq, "length": len(seq), "seconds": elapsed,
        "mean_plddt": 100 * float((out.plddt * atom_mask).sum() / atom_mask.sum()),
        "plddt_scale": "0-100",
        "ptm": float(out.ptm.reshape(-1)[0]),
    }, ca


def pdb_string(model, out):
    # HF 4.57.6 returns pLDDT in [0, 1]; PDB confidence uses [0, 100].
    data = dict(out)
    data["plddt"] = out.plddt * 100
    return model.output_to_pdb(data)[0]


def compare_ca(ca, ref):
    x, y = ca.double(), ref.double()
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vh = torch.linalg.svd(x.T @ y)
    d = torch.eye(3, dtype=x.dtype)
    d[-1, -1] = torch.linalg.det(u @ vh)
    aligned = x @ (u @ d @ vh)
    return float(((aligned - y).square().sum(-1).mean()).sqrt())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence")
    source.add_argument("--fasta", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/esmfold_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--devices", default="0,1", help="Physical chips with fusion=8")
    parser.add_argument("--backbone-layers", type=int, default=36)
    parser.add_argument("--trunk-blocks", type=int, default=0)
    parser.add_argument("--recycles", type=int, default=3, help="Extra passes: 3 gives 4 trunk passes")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Run original CPU FP32 reference first")
    parser.add_argument("--compile-shapes", type=int, default=8, help="Bound compiler tactic search to avoid memory explosion")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/furiosa")
    args = parser.parse_args()
    if not 0 <= args.backbone_layers <= 36 or not 0 <= args.trunk_blocks <= 48:
        parser.error("backbone-layers must be 0..36 and trunk-blocks 0..48")
    if args.recycles < 0 or min(args.threads, args.chunk_size, args.max_length) < 1:
        parser.error("recycles must be >=0; threads, chunk-size and max-length must be positive")
    records = read_sequences(args)
    torch.set_num_threads(args.threads)
    devices = [f"furiosa:{int(i)}" for i in args.devices.split(",")]
    backend, context = None, nullcontext()
    if not args.cpu:
        import furiosa.torch as ft
        ft.set_fusion(8)  # Must precede any device allocation/query.
        compiler_config = ft.config.CompilerConfig()
        compiler_config.max_num_unique_shapes = args.compile_shapes
        compiler_config.local_population_threshold = 2
        compiler_config.expected_total_beam_states = 2000
        compiler_config.tactic_sorting_policy = ft.config.TacticSortingPolicy.ByEstimation
        backend = ft.backend.with_config(compiler_config=compiler_config, eager_fallback=False)
        args.cache.mkdir(parents=True, exist_ok=True)
        context = ft.config.context(compiler_config=compiler_config, cache_dir=str(args.cache))
    args.output.mkdir(parents=True, exist_ok=True)
    log("Loading facebook/esmfold_v1 (CPU FP32)")
    model = EsmForProteinFolding.from_pretrained(args.model, local_files_only=True, low_cpu_mem_usage=True).float().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model.trunk.set_chunk_size(args.chunk_size)
    log(f"Loaded {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B parameters")
    references = {}
    with torch.no_grad():
        if args.validate:
            for index, (name, seq) in enumerate(records):
                log(f"CPU reference: {name} ({len(seq)} aa)")
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                t = time.monotonic()
                out = model(**inputs, num_recycles=args.recycles)
                info, ca = summarize(out, seq, time.monotonic() - t)
                references[index] = (info, ca.clone(), out.plddt.clone())
                (args.output / f"{index:03d}_cpu.pdb").write_text(pdb_string(model, out))
                log(f"CPU reference complete: {info}")
                del out
        with context:
            if not args.cpu:
                prepare(model, args.backbone_layers, args.trunk_blocks, devices, backend)
                log(f"NPU placement: {args.backbone_layers}/36 backbone, {args.trunk_blocks}/48 trunk; devices={devices}")
            for index, (name, seq) in enumerate(records):
                inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False)
                log(f"Folding {name} ({len(seq)} aa)")
                t = time.monotonic()
                out = model(**inputs, num_recycles=args.recycles)
                info, ca = summarize(out, seq, time.monotonic() - t)
                info.update(name=name, mode="cpu" if args.cpu else "rngd-hybrid", recycles=args.recycles,
                            backbone_layers=0 if args.cpu else args.backbone_layers,
                            trunk_blocks=0 if args.cpu else args.trunk_blocks,
                            devices=[] if args.cpu else devices, torch_version=torch.__version__)
                if index in references:
                    ref_info, ref_ca, ref_plddt = references[index]
                    info.update(cpu_reference=ref_info, ca_aligned_rmsd_angstrom=compare_ca(ca, ref_ca),
                                plddt_mean_absolute_difference=100 * float((out.plddt-ref_plddt).abs().mean()))
                info["npu_block_calls"] = {m.label: m.calls for m in model.modules() if isinstance(m, NPUBlock)}
                prefix = args.output / f"{index:03d}"
                prefix.with_suffix(".pdb").write_text(pdb_string(model, out))
                prefix.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
                log(f"Saved {prefix}.pdb: {info}")
                del out


if __name__ == "__main__":
    main()
