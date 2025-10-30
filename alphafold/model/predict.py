# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full AlphaFold protein structure prediction script."""

import enum
import json
import os
import pickle
import time
from typing import Any, Dict

from absl import logging
from alphafold.common import confidence
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.model import model
from alphafold.relax import relax
import jax.numpy as jnp
import numpy as np


@enum.unique
class ModelsToRelax(enum.Enum):
  ALL = 0
  BEST = 1
  NONE = 2


def _jnp_to_np(output: Dict[str, Any]) -> Dict[str, Any]:
  """Recursively changes jax arrays to numpy arrays."""
  for k, v in output.items():
    if isinstance(v, dict):
      output[k] = _jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      output[k] = np.array(v)
  return output


def _save_confidence_json_file(
    plddt: np.ndarray, output_dir: str, model_name: str
) -> None:
  confidence_json = confidence.confidence_json(plddt)

  # Save the confidence json.
  confidence_json_output_path = os.path.join(
      output_dir, f'confidence_{model_name}.json'
  )
  with open(confidence_json_output_path, 'w') as f:
    f.write(confidence_json)


def _save_mmcif_file(
    prot: protein.Protein,
    output_dir: str,
    model_name: str,
    file_id: str,
    model_type: str,
) -> None:
  """Create mmCIF string and save to a file.

  Args:
    prot: Protein object.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
    file_id: The file ID (usually the PDB ID) to be used in the mmCIF.
    model_type: Monomer or multimer.
  """

  mmcif_string = protein.to_mmcif(prot, file_id, model_type)

  # Save the MMCIF.
  mmcif_output_path = os.path.join(output_dir, f'{model_name}.cif')
  with open(mmcif_output_path, 'w') as f:
    f.write(mmcif_string)


def _save_pae_json_file(
    pae: np.ndarray, max_pae: float, output_dir: str, model_name: str
) -> None:
  """Check prediction result for PAE data and save to a JSON file if present.

  Args:
    pae: The n_res x n_res PAE array.
    max_pae: The maximum possible PAE value.
    output_dir: Directory to which files are saved.
    model_name: Name of a model.
  """
  pae_json = confidence.pae_json(pae, max_pae)

  # Save the PAE json.
  pae_json_output_path = os.path.join(output_dir, f'pae_{model_name}.json')
  with open(pae_json_output_path, 'w') as f:
    f.write(pae_json)


def predict_structure(
    fasta_name: str,
    output_dir_base: str,
    feature_dict: pipeline.FeatureDict,
    model_runners: Dict[str, model.RunModel],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    random_seed: int,
    models_to_relax: ModelsToRelax,
    model_type: str,
):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  unrelaxed_pdbs = {}
  unrelaxed_proteins = {}
  relaxed_pdbs = {}
  relax_metrics = {}
  ranking_confidences = {}

  # Run the models.
  num_models = len(model_runners)
  for model_index, (model_name, model_runner) in enumerate(
      model_runners.items()
  ):
    logging.info('Running model %s on %s', model_name, fasta_name)
    t_0 = time.time()
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed
    )
    timings[f'process_features_{model_name}'] = time.time() - t_0

    t_0 = time.time()
    prediction_result = model_runner.predict(
        processed_feature_dict, random_seed=model_random_seed
    )
    t_diff = time.time() - t_0
    timings[f'predict_and_compile_{model_name}'] = t_diff
    logging.info(
        'Total JAX model %s on %s predict time (includes compilation time, see'
        ' --benchmark): %.1fs',
        model_name,
        fasta_name,
        t_diff,
    )
    num_recycles = prediction_result['num_recycles'].item()
    timings[f'num_recycles_{model_name}'] = num_recycles

    if benchmark:
      t_0 = time.time()
      model_runner.predict(
          processed_feature_dict, random_seed=model_random_seed
      )
      t_diff = time.time() - t_0
      timings[f'predict_benchmark_{model_name}'] = t_diff
      logging.info(
          'Total JAX model %s on %s predict time (excludes compilation time):'
          ' %.1fs',
          model_name,
          fasta_name,
          t_diff,
      )

    plddt = prediction_result['plddt']
    _save_confidence_json_file(plddt, output_dir, model_name)
    ranking_confidences[model_name] = prediction_result['ranking_confidence']

    if (
        'predicted_aligned_error' in prediction_result
        and 'max_predicted_aligned_error' in prediction_result
    ):
      pae = prediction_result['predicted_aligned_error']
      max_pae = prediction_result['max_predicted_aligned_error']
      _save_pae_json_file(pae, float(max_pae), output_dir, model_name)

    # Remove jax dependency from results.
    np_prediction_result = _jnp_to_np(dict(prediction_result))

    # Save the model outputs.
    result_output_path = os.path.join(output_dir, f'result_{model_name}.pkl')
    with open(result_output_path, 'wb') as f:
      pickle.dump(np_prediction_result, f, protocol=4)

    # Add the predicted LDDT in the b-factor column.
    # Note that higher predicted LDDT value means higher model confidence.
    plddt_b_factors = np.repeat(
        plddt[:, None], residue_constants.atom_type_num, axis=-1
    )
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=prediction_result,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=not model_runner.multimer_mode,
    )

    unrelaxed_proteins[model_name] = unrelaxed_protein
    unrelaxed_pdbs[model_name] = protein.to_pdb(unrelaxed_protein)
    unrelaxed_pdb_path = os.path.join(output_dir, f'unrelaxed_{model_name}.pdb')
    with open(unrelaxed_pdb_path, 'w') as f:
      f.write(unrelaxed_pdbs[model_name])

    _save_mmcif_file(
        prot=unrelaxed_protein,
        output_dir=output_dir,
        model_name=f'unrelaxed_{model_name}',
        file_id=str(model_index),
        model_type=model_type,
    )

  # Rank by model confidence.
  ranked_order = [
      model_name
      for model_name, confidence in sorted(
          ranking_confidences.items(), key=lambda x: x[1], reverse=True
      )
  ]

  # Relax predictions.
  if models_to_relax == ModelsToRelax.BEST:
    to_relax = [ranked_order[0]]
  elif models_to_relax == ModelsToRelax.ALL:
    to_relax = ranked_order
  elif models_to_relax == ModelsToRelax.NONE:
    to_relax = []
  else:
    raise ValueError(f'Unrecognized models_to_relax value {models_to_relax}')

  for model_name in to_relax:
    t_0 = time.time()
    relaxed_pdb_str, _, violations = amber_relaxer.process(
        prot=unrelaxed_proteins[model_name]
    )
    relax_metrics[model_name] = {
        'remaining_violations': violations,
        'remaining_violations_count': sum(violations),
    }
    timings[f'relax_{model_name}'] = time.time() - t_0

    relaxed_pdbs[model_name] = relaxed_pdb_str

    # Save the relaxed PDB.
    relaxed_output_path = os.path.join(output_dir, f'relaxed_{model_name}.pdb')
    with open(relaxed_output_path, 'w') as f:
      f.write(relaxed_pdb_str)

    relaxed_protein = protein.from_pdb_string(relaxed_pdb_str)
    _save_mmcif_file(
        prot=relaxed_protein,
        output_dir=output_dir,
        model_name=f'relaxed_{model_name}',
        file_id='0',
        model_type=model_type,
    )

  # Write out relaxed PDBs in rank order.
  for idx, model_name in enumerate(ranked_order):
    ranked_output_path = os.path.join(output_dir, f'ranked_{idx}.pdb')
    with open(ranked_output_path, 'w') as f:
      if model_name in relaxed_pdbs:
        f.write(relaxed_pdbs[model_name])
      else:
        f.write(unrelaxed_pdbs[model_name])

    if model_name in relaxed_pdbs:
      protein_instance = protein.from_pdb_string(relaxed_pdbs[model_name])
    else:
      protein_instance = protein.from_pdb_string(unrelaxed_pdbs[model_name])

    _save_mmcif_file(
        prot=protein_instance,
        output_dir=output_dir,
        model_name=f'ranked_{idx}',
        file_id=str(idx),
        model_type=model_type,
    )

  ranking_output_path = os.path.join(output_dir, 'ranking_debug.json')
  with open(ranking_output_path, 'w') as f:
    label = 'iptm+ptm' if 'iptm' in prediction_result else 'plddts'
    f.write(
        json.dumps(
            {label: ranking_confidences, 'order': ranked_order}, indent=4
        )
    )

  if models_to_relax != ModelsToRelax.NONE:
    relax_metrics_path = os.path.join(output_dir, 'relax_metrics.json')
    with open(relax_metrics_path, 'w') as f:
      f.write(json.dumps(relax_metrics, indent=4))
  return timings
