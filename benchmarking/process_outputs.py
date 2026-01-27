import argparse
import json
import pathlib

from Bio import SeqIO
import pandas as pd
from tqdm import tqdm


def collect_residue_counts(fasta_dir: str | pathlib.Path) -> dict[str, int]:
  fasta_dir = pathlib.Path(fasta_dir)

  return {
      fasta.stem: sum(len(r) for r in SeqIO.parse(fasta, format="fasta"))
      for fasta in tqdm(list(fasta_dir.glob("*.fasta")), desc="Reading fasta files for residue counts")
  }


def read_json(json_file: pathlib.Path) -> dict:
  try:
    with open(json_file, "r") as f:
      data = json.load(f)
  except json.JSONDecodeError as e:
    raise RuntimeError(f"Error parsing {json_file}: {e}")
  return data


def collect_run_data(base_path: str | pathlib.Path) -> pd.DataFrame:
  """
  Collect output from alphafold runs into a pandas DataFrame.

  This is designed so that it can work regardless of the model hierarchy used
  (e.g. separate subdirectories for different seeds or model checkpoints).

  Args:
      base_path: Path to the base output directory containing subdirectories
          for each protein complex

  Returns:
      DataFrame with columns:
          - complex
          - model_family
          - model_name
          - seed
          - compile_time
          - run_time
          - recycle_count
          - relaxed_dockq
          - unrelaxed_dockq
          - confidence
  """
  base_dir = pathlib.Path(base_path)

  if not base_dir.exists():
    raise FileNotFoundError(f"Directory not found: {base_path}")

  data_records = []

  for complex_dir in tqdm(sorted(base_dir.iterdir()), desc="Parsing run data"):
    if not complex_dir.is_dir():
      continue

    complex = complex_dir.name

    for seed_file in complex_dir.glob("**/random_seeds_debug.json"):
      timings_file = seed_file.with_name("timings.json")
      ranking_file = seed_file.with_name("ranking_debug.json")

      seeds = read_json(seed_file)
      timings = read_json(timings_file)
      confidences = read_json(ranking_file)["iptm+ptm"]

      for model_pred, seed in seeds.items():
        try:
          model_name = model_pred.removesuffix("_pred_0")
          compile_key = f"predict_and_compile_{model_pred}"
          benchmark_key = f"predict_benchmark_{model_pred}"
          recycles_key = f"num_recycles_{model_pred}"

          predict_and_compile = timings[compile_key]
          predict_benchmark = timings[benchmark_key]
          compile_time = predict_and_compile - predict_benchmark

          relaxed_dockq_file = seed_file.with_name(
              f"relaxed_{model_pred}_dockq.json"
          )
          unrelaxed_dockq_file = seed_file.with_name(
              f"unrelaxed_{model_pred}_dockq.json"
          )

          relaxed_dockq = read_json(relaxed_dockq_file)["GlobalDockQ"]
          unrelaxed_dockq = read_json(unrelaxed_dockq_file)["GlobalDockQ"]

          record = {
              "complex": complex,
              "model_family": "alphafold2",
              "model_name": model_name,
              "seed": seed,
              "compile_time": compile_time,
              "run_time": predict_benchmark,
              "recycle_count": timings[recycles_key],
              "relaxed_dockq": relaxed_dockq,
              "unrelaxed_dockq": unrelaxed_dockq,
              "confidence": confidences[model_pred],
          }
          data_records.append(record)
        except Exception as e:
          raise RuntimeError(
              f"Error processing {complex=}, {model_pred=}, {seed=}"
          ) from e

  df = pd.DataFrame(data_records)

  df = df.sort_values(["complex", "seed", "model_name"]).reset_index(drop=True)

  return df


def create_df(
    run_dir: str | pathlib.Path, fasta_dir: str | pathlib.Path
) -> pd.DataFrame:
  df = collect_run_data(run_dir)

  residue_counts = collect_residue_counts(fasta_dir)

  df["residue_count"] = df["complex"].map(residue_counts)

  return df


def main(args):
  df = create_df(args.run_dir, args.fasta_dir)
  output_file = pathlib.Path(args.output_file)
  output_file.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(output_file, index=False)
  print(f"DataFrame saved to {output_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Process benchmarking outputs into a DataFrame."
  )
  parser.add_argument(
      "--run_dir",
      type=str,
      help=(
          "Path to the base output directory containing subdirectories for each"
          " protein complex."
      ),
  )
  parser.add_argument(
      "--fasta_dir",
      type=str,
      help="Path to the directory containing FASTA files.",
  )
  parser.add_argument(
      "--output_file",
      type=str,
      required=False,
      help=(
          "Path to save the output CSV file. Defaults to"
          " <run_dir>/processed_results.csv"
      ),
  )

  args = parser.parse_args()

  args.run_dir = pathlib.Path(args.run_dir)
  args.fasta_dir = pathlib.Path(args.fasta_dir)

  if not args.run_dir.exists():
    raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
  if not args.run_dir.is_dir():
    raise NotADirectoryError(
        f"Run directory is not a directory: {args.run_dir}"
    )
  if not args.fasta_dir.exists():
    raise FileNotFoundError(f"FASTA directory not found: {args.fasta_dir}")
  if not args.fasta_dir.is_dir():
    raise NotADirectoryError(
        f"FASTA directory is not a directory: {args.fasta_dir}"
    )

  args.fasta_dir = pathlib.Path(args.fasta_dir)

  if args.output_file is None:
    args.output_file = args.run_dir / "processed_results.csv"

  args.output_file = pathlib.Path(args.output_file)
  main(args)
