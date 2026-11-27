"""Numerical checks against the installed, unmodified HF implementation."""
import copy
import types
import unittest
import torch
from transformers import EsmConfig, EsmModel, EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import EsmFoldSelfAttention
from esmfold_rngd import attention_forward, rotary_forward, fold_attention_forward, language_representations, compare_ca

class CompatibilityTest(unittest.TestCase):
    def test_language_states_match_original_with_padding(self):
        torch.manual_seed(17)
        config = EsmConfig(vocab_size=33, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=2, num_attention_heads=4,
                           pad_token_id=1, mask_token_id=32, position_embedding_type='rotary',
                           token_dropout=True, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
        config._attn_implementation = 'eager'
        backbone = EsmModel(config, add_pooling_layer=False).eval()
        holder = types.SimpleNamespace(esm=backbone, parameters=backbone.parameters,
                                       config=types.SimpleNamespace(esmfold_config=types.SimpleNamespace(bypass_lm=False)),
                                       esm_dict_cls_idx=0, esm_dict_padding_idx=1, esm_dict_eos_idx=2)
        tokens = torch.tensor([[4, 5, 6, 7], [8, 9, 1, 1]])
        with torch.no_grad():
            expected = EsmForProteinFolding.compute_language_model_representations(holder, tokens)
            for layer in backbone.encoder.layer:
                attn = layer.attention.self
                attn.forward = types.MethodType(attention_forward, attn)
                attn.rotary_embeddings.forward = types.MethodType(rotary_forward, attn.rotary_embeddings)
            actual = language_representations(holder, tokens)
        self.assertEqual(actual.shape, (2, 4, 3, 64))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_fold_attention_matches_original(self):
        torch.manual_seed(18)
        original = EsmFoldSelfAttention(64, 4, 16, gated=True).eval()
        patched = copy.deepcopy(original)
        patched.forward = types.MethodType(fold_attention_forward, patched)
        x, bias = torch.randn(2, 7, 64), torch.randn(2, 7, 7, 4)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
        with torch.no_grad():
            expected = original(x, mask=mask, bias=bias)
            actual = patched(x, mask=mask, bias=bias)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_rmsd_removes_rigid_rotation_translation(self):
        x = torch.randn(20, 3, dtype=torch.float64)
        rotation = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]], dtype=x.dtype)
        self.assertLess(compare_ca(x @ rotation + 10, x), 1e-10)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
