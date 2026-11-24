import numpy as np
import torch

from stgnn.metrics import paired_test, section_macro_f1
from stgnn.models import build_model
from stgnn.train import class_weights, stack_sections


def test_stack_sections_offsets_and_symmetrises_edges():
    b = stack_sections([np.zeros((3, 2)), np.zeros((2, 2))],
                       [np.array([[0, 1], [1, 2]]), np.array([[0, 1]])],
                       [np.zeros(3, int), np.ones(2, int)])
    pairs = set(map(tuple, b.edge_index.T.tolist()))
    assert pairs == {(0, 1), (1, 0), (1, 2), (2, 1), (3, 4), (4, 3)}
    assert b.section_ptr.tolist() == [0, 3, 5]


def test_models_shapes_and_no_edge_isolation():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    ei = torch.cat([ei, ei.flip(0)], 1)
    empty = torch.zeros((2, 0), dtype=torch.long)
    for arch in ["mlp", "sage", "gcn"]:
        m = build_model(arch, 8, 4).eval()
        assert m(x, ei).shape == (10, 4)
        # with no edges, a spot's output must not depend on any other spot
        out1 = m(x, empty)
        x2 = x.clone()
        x2[1:] = torch.randn(9, 8)
        assert torch.allclose(m(x2, empty)[0], out1[0], atol=1e-6)
    # with edges, the graph models do use neighbours
    m = build_model("sage", 8, 4).eval()
    x2 = x.clone()
    x2[1] += 5
    assert not torch.allclose(m(x2, ei)[0], m(x, ei)[0])


def test_section_macro_f1_ignores_absent_layers():
    y = np.array(["L3", "L3", "WM", "WM"])
    assert section_macro_f1(y, y) == 1.0
    # predicting an absent layer is still an error for the true layer
    p = np.array(["L1", "L3", "WM", "WM"])
    f1_l3 = 2 * 1 * 0.5 / 1.5
    assert np.isclose(section_macro_f1(y, p), (f1_l3 + 1.0) / 2)


def test_class_weights_zero_for_missing_class():
    w = class_weights(torch.tensor([0, 0, 0, 1]), 3)
    assert w[2] == 0 and w[1] > w[0]


def test_paired_test():
    r = paired_test(np.arange(12) + 1.0, np.arange(12) + 0.0)
    assert r["mean_diff"] == 1.0 and r["n_a_better"] == 12 and r["wilcoxon_p"] < 0.001
