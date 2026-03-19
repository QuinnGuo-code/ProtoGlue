"""Run ProtoGlue on Human Breast Cancer dataset (RNA + ADT)."""

from pathlib import Path
import protoglue as pg

result = pg.run_pipeline(
    data_dir=Path("data/human_breast_cancer"),
    output_dir=Path("results/human_breast_cancer"),
    config={"K_GRID": list(range(5, 10))},
    device="cuda",
    seed=2022,
    auto_k=True,
    verbose=True,
)

print(f"Clusters: {result['metrics']['n_clusters']}")
print(f"Spatial purity: {result['metrics']['spatial_edge_purity']:.4f}")
