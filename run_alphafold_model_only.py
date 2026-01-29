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
import pathlib
import pickle
import random
import sys
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.model import predict
from alphafold.relax import relax
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'feature_paths',
    None,
    'Paths to features.npz files with precomputed features. Paths should be'
    ' separated by commas. All FASTA paths must have a unique basename as the'
    ' basename is used to name the output directories for each prediction.'
    ' These should generally match the basename of the FASTA file that was used'
    ' to create the features.',
)

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string(
    'output_dir', None, 'Path to a directory that will store the results.'
)
flags.DEFINE_enum(
    'model_preset',
    'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Choose preset model configuration - the monomer model, '
    'the monomer model with extra ensembling, monomer model with '
    'pTM head, or multimer model',
)
flags.DEFINE_integer(
    'num_recycle',
    None,
    'The number of recycle steps to perform. By default, uses the setting from'
    ' the model configuration.',
)
flags.DEFINE_float(
    'recycle_early_stop_tolerance',
    None,
    'A positive value will stop prediction early if the difference in pairwise'
    ' distances between recycling steps is less than the tolerance. A negative'
    ' value will disable early stopping, i.e. the model will always run'
    ' `num_recycle` number of recycling iterations.',
)
flags.DEFINE_list(
    'model_names',
    None,
    'Names of models to use. If not set, all models for the preset will be'
    ' used.',
)
flags.DEFINE_boolean(
    'benchmark',
    False,
    'Run multiple JAX model evaluations '
    'to obtain a timing that excludes the compilation time, '
    'which should be more indicative of the time required for '
    'inferencing many proteins.',
)
flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed for the data '
    'pipeline. By default, this is randomly generated. Note '
    'that even if this is set, Alphafold may still not be '
    'deterministic, because processes like GPU inference are '
    'nondeterministic.',
)
flags.DEFINE_boolean(
    'consistent_random_seeds',
    None,
    'By default, each model '
    'uses a different random seed based on the one set with '
    '--random_seed. If this is set, they will instead all be '
    'initialized with the same random seed. If '
    '--num_multimer_predictions_per_model is greater than 1, '
    'a different random seed will still be used for subsequent '
    'predictions, but the nth prediction will always use the '
    'same random seed regardless of how many predictions are '
    'made.',
)
flags.DEFINE_integer(
    'num_multimer_predictions_per_model',
    5,
    'How many '
    'predictions (each with a different random seed) will be '
    'generated per model. E.g. if this is 2 and there are 5 '
    'models then there will be 10 predictions per input. '
    'Note: this FLAG only applies if model_preset=multimer',
)
flags.DEFINE_bool(
    'clear_cache',
    False,
    'Whether to clear the JAX compilation cache between models. '
    'This can reduce memory usage but may increase runtime.',
)
flags.DEFINE_enum_class(
    'models_to_relax',
    predict.ModelsToRelax.BEST,
    predict.ModelsToRelax,
    'The models to run the final relaxation step on. '
    'If `all`, all models are relaxed, which may be time '
    'consuming. If `best`, only the most confident model '
    'is relaxed. If `none`, relaxation is not run. Turning '
    'off relaxation might result in predictions with '
    'distracting stereochemical violations but might help '
    'in case you are having issues with the relaxation '
    'stage.',
)
flags.DEFINE_boolean(
    'use_gpu_relax',
    None,
    'Whether to relax on GPU. '
    'Relax on GPU can be much faster than CPU, so it is '
    'recommended to enable if possible. GPUs must be available'
    ' if this setting is enabled.',
)
flags.DEFINE_boolean(
    'save_full_results',
    True,
    'Whether to save the full pickled prediction results. These have'
    ' interesting data, but they are quite large.',
)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def predict_structures(
    feature_path: str,
    system_name: str,
    output_dir_base: str,
    model_runners: Dict[str, tuple[int, model.RunModel]],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    clear_cache: bool,
    models_to_relax: predict.ModelsToRelax,
    model_type: str,
):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', system_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, system_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if not os.path.exists(feature_path):
    raise ValueError('Feature file %s does not exist' % feature_path)

  feature_dict = None
  logging.info('Reading features from %s', feature_path)
  with np.load(feature_path, allow_pickle=False) as data:
    feature_dict = {k: data[k] for k in data.files}

  random_seeds_output_path = os.path.join(output_dir, 'random_seeds_debug.json')
  with open(random_seeds_output_path, 'w') as f:
    f.write(
        json.dumps(
            {name: seed for name, (seed, _) in model_runners.items()}, indent=4
        )
    )

  timings = predict.predict_structure(
      fasta_name=system_name,
      output_dir_base=output_dir_base,
      feature_dict=feature_dict,
      model_runners=model_runners,
      amber_relaxer=amber_relaxer,
      benchmark=benchmark,
      clear_cache=clear_cache,
      models_to_relax=models_to_relax,
      model_type=model_type,
      save_full_results=FLAGS.save_full_results,
  )

  logging.info('Final timings for %s: %s', system_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_multimer_system = 'multimer' in FLAGS.model_preset
  model_type = 'Multimer' if run_multimer_system else 'Monomer'

  # Check for duplicate FASTA file names.
  system_names = [pathlib.Path(p).stem for p in FLAGS.feature_paths]
  if len(system_names) != len(set(system_names)):
    raise ValueError('All feature paths must have a unique basename.')

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
  else:
    num_predictions_per_model = 1

  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  if FLAGS.model_names is not None:
    model_names = [m for m in model_names if m in FLAGS.model_names]

    if len(model_names) != len(FLAGS.model_names):
      invalid_models = set(FLAGS.model_names) - set(model_names)
      logging.error(
          'Invalid model names: %s. Valid names are: %s',
          ', '.join(invalid_models),
          ', '.join(config.MODEL_PRESETS[FLAGS.model_preset]),
      )
      raise ValueError('Some model names in --model_names are not valid.')

  num_total_model_predictions = len(model_names) * num_predictions_per_model

  random_seed = FLAGS.random_seed
  if FLAGS.consistent_random_seeds:
    max_seed_size = sys.maxsize - num_predictions_per_model
    if random_seed is None:
      random_seed = random.randrange(max_seed_size)
    elif random_seed > max_seed_size:
      raise ValueError(
          'Random seed %d is larger than maximum seed size %d. '
          'It must be less than sys.maxsize (%d) - '
          'predictions per model (%d)',
          random_seed,
          max_seed_size,
          sys.maxsize,
          num_predictions_per_model,
      )
    model_random_seeds = list(
        range(random_seed, random_seed + num_predictions_per_model)
    ) * len(model_names)
  else:
    # For historical reasons/backward compatibility, this is the maximum allowable size
    max_seed_size = sys.maxsize // num_total_model_predictions
    if random_seed is None:
      random_seed = random.randrange(max_seed_size)
    elif random_seed > max_seed_size:
      raise ValueError(
          'Random seed %d is larger than maximum seed size %d. '
          'It must be less than sys.maxsize (%d) // '
          'total number of model predictions (%d)',
          random_seed,
          max_seed_size,
          sys.maxsize,
          num_total_model_predictions,
      )
    model_random_seeds = list(
        range(random_seed, random_seed + num_total_model_predictions)
    )

  logging.info('Using random seed %d', random_seed)
  assert len(model_random_seeds) == num_total_model_predictions, (
      'Should have same number of model random seeds'
      f' ({len(model_random_seeds)}) as model predictions'
      f' ({num_total_model_predictions})'
  )

  model_runners = {}
  for i, model_name in enumerate(model_names):
    model_config = config.model_config(model_name)
    if FLAGS.num_recycle is not None:
      model_config.model.num_recycle = FLAGS.num_recycle
    if FLAGS.recycle_early_stop_tolerance is not None:
      model_config.model.recycle_early_stop_tolerance = (
          FLAGS.recycle_early_stop_tolerance
      )
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir
    )
    model_runner = model.RunModel(model_config, model_params)
    for j in range(num_predictions_per_model):
      model_index = i * num_predictions_per_model + j
      model_random_seed = model_random_seeds[model_index]
      model_runners[f'{model_name}_pred_{j}'] = (
          model_random_seed,
          model_runner,
      )

  logging.info(
      'Have %d model predictions: %s',
      len(model_runners),
      list(model_runners.keys()),
  )
  assert len(model_runners) == num_total_model_predictions, (
      f'Should have same number of model runners ({len(model_runners)}) '
      f'as model predictions ({num_total_model_predictions})'
  )

  amber_relaxer = relax.AmberRelaxation(
      max_iterations=RELAX_MAX_ITERATIONS,
      tolerance=RELAX_ENERGY_TOLERANCE,
      stiffness=RELAX_STIFFNESS,
      exclude_residues=RELAX_EXCLUDE_RESIDUES,
      max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
      use_gpu=FLAGS.use_gpu_relax,
  )

  # Predict structure for each of the sequences.
  for i, feature_path in enumerate(FLAGS.feature_paths):
    predict_structures(
        feature_path=feature_path,
        system_name=system_names[i],
        output_dir_base=FLAGS.output_dir,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        clear_cache=FLAGS.clear_cache,
        models_to_relax=FLAGS.models_to_relax,
        model_type=model_type,
    )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'feature_paths',
      'output_dir',
      'data_dir',
      'use_gpu_relax',
  ])

  app.run(main)
