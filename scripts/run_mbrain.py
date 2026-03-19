"""Run ProtoGlue on Mouse Brain dataset (RNA + ATAC)."""

from pathlib import Path
import protoglue as pg

result = pg.run_pipeline(
    data_dir=Path("data/mouse_brain"),
    output_dir=Path("results/mouse_brain"),
    config={"K_GRID": list(range(6, 11))},
    device="cuda",
    seed=2022,
    auto_k=True,
    verbose=True,
)

print(f"Clusters: {result['metrics']['n_clusters']}")
print(f"Spatial purity: {result['metrics']['spatial_edge_purity']:.4f}")
