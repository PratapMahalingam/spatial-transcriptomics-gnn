from stgnn.experiment import Config, run_cv, summarise
from stgnn.synthetic import make_synthetic


def test_synthetic_pipeline_runs_and_reports_all_comparisons():
    secs = make_synthetic(n_donors=2, sections_per_donor=2, n_rows=14, n_cols=14, n_genes=60, signal=0.3)
    cfg = Config(n_hvg=60, n_pcs=10, seeds=(0,), max_epochs=15, patience=5, swaps_per_edge=2, log=False)
    res, g, preds, classes = run_cv(secs, cfg)
    assert set(res.variant) == set(cfg.variants)
    assert res.section.nunique() == 4
    s = summarise(res, g)
    assert len(s["comparisons"]) == 7
    assert s["graph"]["homophily_real_mean"] > s["graph"]["homophily_rewired_mean"]
    assert set(preds.section) == {x.name for x in secs}
