from pathlib import Path
import argparse
import protoglue as pg


def main():
    parser = argparse.ArgumentParser(description="ProtoGlue minimal example")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/human_lymph_node",
        help="Path to directory containing at least two .h5ad files (RNA + second modality)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/minimal_example",
        help="Path to output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cuda' or 'cpu'"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please prepare example data first. See data/README.md for details."
        )

    h5ad_files = list(data_dir.glob("*.h5ad"))
    if len(h5ad_files) < 2:
        raise FileNotFoundError(
            f"Found {len(h5ad_files)} .h5ad file(s) in {data_dir}, but ProtoGlue requires at least 2:\n"
            f"1 RNA file + 1 second-modality file.\n"
            f"Please see data/README.md for the expected data structure."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1/6: Loading and preprocessing data...")

    result = pg.run_pipeline(
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        device=args.device
    )

    print("=" * 60)
    print("Finished.")
    print("Output saved to:", output_dir)
    print("Result keys:", result.keys() if isinstance(result, dict) else result)


if __name__ == "__main__":
    main()
