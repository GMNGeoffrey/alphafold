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
import shutil
import sys
import time
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.model import predict
from alphafold.relax import relax
import numpy as np

# Internal import (7716).

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths',
    None,
    'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.',
)

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string(
    'output_dir', None, 'Path to a directory that will store the results.'
)
flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the JackHMMER executable.',
)
flags.DEFINE_string(
    'hhblits_binary_path',
    shutil.which('hhblits'),
    'Path to the HHblits executable.',
)
flags.DEFINE_string(
    'hhsearch_binary_path',
    shutil.which('hhsearch'),
    'Path to the HHsearch executable.',
)
flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the hmmsearch executable.',
)
flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the hmmbuild executable.',
)
flags.DEFINE_string(
    'kalign_binary_path',
    shutil.which('kalign'),
    'Path to the Kalign executable.',
)
flags.DEFINE_string(
    'uniref90_database_path',
    None,
    'Path to the Uniref90 database for use by JackHMMER.',
)
flags.DEFINE_string(
    'mgnify_database_path',
    None,
    'Path to the MGnify database for use by JackHMMER.',
)
flags.DEFINE_string(
    'bfd_database_path', None, 'Path to the BFD database for use by HHblits.'
)
flags.DEFINE_string(
    'small_bfd_database_path',
    None,
    'Path to the small version of BFD used with the "reduced_dbs" preset.',
)
flags.DEFINE_string(
    'uniref30_database_path',
    None,
    'Path to the UniRef30 database for use by HHblits.',
)
flags.DEFINE_string(
    'uniprot_database_path',
    None,
    'Path to the Uniprot database for use by JackHMMer.',
)
flags.DEFINE_string(
    'pdb70_database_path',
    None,
    'Path to the PDB70 database for use by HHsearch.',
)
flags.DEFINE_string(
    'pdb_seqres_database_path',
    None,
    'Full filepath to the '
    'PDB seqres database file (not just the directory) for use '
    'by hmmsearch.',
)
flags.DEFINE_string(
    'template_mmcif_dir',
    None,
    'Path to a directory with '
    'template mmCIF structures, each named <pdb_id>.cif',
)
flags.DEFINE_string(
    'max_template_date',
    None,
    'Maximum template release date '
    'to consider. Important if folding historical test sets.',
)
flags.DEFINE_string(
    'obsolete_pdbs_path',
    None,
    'Path to file containing a '
    'mapping from obsolete PDB IDs to the PDB IDs of their '
    'replacements.',
)
flags.DEFINE_enum(
    'db_preset',
    'full_dbs',
    ['full_dbs', 'reduced_dbs'],
    'Choose preset MSA database configuration - '
    'smaller genetic database config (reduced_dbs) or '
    'full genetic database config  (full_dbs)',
)
flags.DEFINE_enum(
    'model_preset',
    'monomer',
    ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
    'Choose preset model configuration - the monomer model, '
    'the monomer model with extra ensembling, monomer model with '
    'pTM head, or multimer model',
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
    'Whether to clear the JAX compilation cache between predictions. '
    'This can reduce memory usage but may increase runtime.',
)
flags.DEFINE_boolean(
    'use_precomputed_msas',
    False,
    'Whether to read MSAs that '
    'have been written to disk instead of running the MSA '
    'tools. The MSA files are looked up in the output '
    'directory, so it must stay the same between multiple '
    'runs that are to reuse the MSAs. WARNING: This will not '
    'check if the sequence, database or configuration have '
    'changed.',
)
flags.DEFINE_boolean(
    'use_precomputed_features', False, 'Whether to use existing features.pkl'
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
flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for Jackhmmer. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
flags.DEFINE_integer(
    'hmmsearch_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for HMMsearch. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)
flags.DEFINE_integer(
    'hhsearch_n_cpu',
    # Unfortunately, os.process_cpu_count() is only available in Python 3.13+.
    min(len(os.sched_getaffinity(0)), 8),
    'Number of CPUs to use for HHsearch. Defaults to min(cpu_count, 8). Going'
    ' above 8 CPUs provides very little additional speedup.',
    lower_bound=0,
)

FLAGS = flags.FLAGS

MAX_TEMPLATE_HITS = 20
RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def _check_flag(flag_name: str, other_flag_name: str, should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(
        f'{flag_name} must {verb} set when running with '
        f'"--{other_flag_name}={FLAGS[other_flag_name].value}".'
    )


def search_and_predict(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    data_pipeline: Union[pipeline.DataPipeline, pipeline_multimer.DataPipeline],
    model_runners: Dict[str, tuple[int, model.RunModel]],
    amber_relaxer: relax.AmberRelaxation,
    benchmark: bool,
    clear_cache: bool,
    models_to_relax: predict.ModelsToRelax,
    use_precomputed_features: bool,
    model_type: str,
):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)
  timings = {}
  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  msa_output_dir = os.path.join(output_dir, 'msas')
  if not os.path.exists(msa_output_dir):
    os.makedirs(msa_output_dir)

  features_output_pkl_path = os.path.join(output_dir, 'features.pkl')
  features_output_npz_path = os.path.join(output_dir, 'features.npz')

  feature_dict = None
  if use_precomputed_features:
    if os.path.exists(features_output_npz_path):
      logging.info('Reading features from %s', features_output_npz_path)
      with np.load(features_output_npz_path, allow_pickle=False) as data:
        feature_dict = {k: data[k] for k in data.files}
    # Fall back to older pickle format
    elif os.path.exists(features_output_pkl_path):
      logging.info('Reading features from %s', features_output_pkl_path)
      with open(features_output_pkl_path, 'rb') as f:
        feature_dict = pickle.load(f)
    else:
      logging.warning(
          'use_precomputed_features is set but neither %s nor %s exist. Running'
          ' full feature pipeline',
          features_output_npz_path,
          features_output_pkl_path,
      )

  if feature_dict is None:
    # Get features.
    t_0 = time.time()
    feature_dict = data_pipeline.process(
        input_fasta_path=fasta_path, msa_output_dir=msa_output_dir
    )
    timings['features'] = time.time() - t_0

    logging.info('Writing features to %s', features_output_npz_path)
    np.savez(features_output_npz_path, **feature_dict)

    # For backward compatibility, also write out features as a pickled dictionary.
    logging.info('Writing features to %s', features_output_pkl_path)
    with open(features_output_pkl_path, 'wb') as f:
      pickle.dump(feature_dict, f, protocol=4)

  random_seeds_output_path = os.path.join(output_dir, 'random_seeds_debug.json')
  with open(random_seeds_output_path, 'w') as f:
    f.write(
        json.dumps(
            {name: seed for name, (seed, _) in model_runners.items()}, indent=4
        )
    )

  if model_runners:
    timings.update(
        predict.predict_structure(
            fasta_name=fasta_name,
            output_dir_base=output_dir_base,
            feature_dict=feature_dict,
            model_runners=model_runners,
            amber_relaxer=amber_relaxer,
            benchmark=benchmark,
            clear_cache=clear_cache,
            models_to_relax=models_to_relax,
            model_type=model_type,
        )
    )

  logging.info('Final timings for %s: %s', fasta_name, timings)

  timings_output_path = os.path.join(output_dir, 'timings.json')
  with open(timings_output_path, 'w') as f:
    f.write(json.dumps(timings, indent=4))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  for tool_name in (
      'jackhmmer',
      'hhblits',
      'hhsearch',
      'hmmsearch',
      'hmmbuild',
      'kalign',
  ):
    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(
          f'Could not find path to the "{tool_name}" binary. Make '
          'sure it is installed on your system.'
      )

  use_small_bfd = FLAGS.db_preset == 'reduced_dbs'
  _check_flag(
      'small_bfd_database_path', 'db_preset', should_be_set=use_small_bfd
  )
  _check_flag('bfd_database_path', 'db_preset', should_be_set=not use_small_bfd)
  _check_flag(
      'uniref30_database_path', 'db_preset', should_be_set=not use_small_bfd
  )

  run_multimer_system = 'multimer' in FLAGS.model_preset
  model_type = 'Multimer' if run_multimer_system else 'Monomer'
  _check_flag(
      'pdb70_database_path',
      'model_preset',
      should_be_set=not run_multimer_system,
  )
  _check_flag(
      'pdb_seqres_database_path',
      'model_preset',
      should_be_set=run_multimer_system,
  )
  _check_flag(
      'uniprot_database_path', 'model_preset', should_be_set=run_multimer_system
  )

  if FLAGS.model_preset == 'monomer_casp14':
    num_ensemble = 8
  else:
    num_ensemble = 1

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  if run_multimer_system:
    template_searcher = hmmsearch.Hmmsearch(
        binary_path=FLAGS.hmmsearch_binary_path,
        hmmbuild_binary_path=FLAGS.hmmbuild_binary_path,
        database_path=FLAGS.pdb_seqres_database_path,
        cpu=FLAGS.hmmsearch_n_cpu,
    )
    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
    )
  else:
    template_searcher = hhsearch.HHSearch(
        binary_path=FLAGS.hhsearch_binary_path,
        databases=[FLAGS.pdb70_database_path],
        cpu=FLAGS.hhsearch_n_cpu,
    )
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=FLAGS.template_mmcif_dir,
        max_template_date=FLAGS.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=FLAGS.kalign_binary_path,
        release_dates_path=None,
        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path,
    )

  monomer_data_pipeline = pipeline.DataPipeline(
      jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
      hhblits_binary_path=FLAGS.hhblits_binary_path,
      uniref90_database_path=FLAGS.uniref90_database_path,
      mgnify_database_path=FLAGS.mgnify_database_path,
      bfd_database_path=FLAGS.bfd_database_path,
      uniref30_database_path=FLAGS.uniref30_database_path,
      small_bfd_database_path=FLAGS.small_bfd_database_path,
      template_searcher=template_searcher,
      template_featurizer=template_featurizer,
      use_small_bfd=use_small_bfd,
      use_precomputed_msas=FLAGS.use_precomputed_msas,
      msa_tools_n_cpu=FLAGS.jackhmmer_n_cpu,
  )

  if run_multimer_system:
    num_predictions_per_model = FLAGS.num_multimer_predictions_per_model
    data_pipeline = pipeline_multimer.DataPipeline(
        monomer_data_pipeline=monomer_data_pipeline,
        jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
        uniprot_database_path=FLAGS.uniprot_database_path,
        use_precomputed_msas=FLAGS.use_precomputed_msas,
        jackhmmer_n_cpu=FLAGS.jackhmmer_n_cpu,
    )
  else:
    num_predictions_per_model = 1
    data_pipeline = monomer_data_pipeline

  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
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
    if run_multimer_system:
      model_config.model.num_ensemble_eval = num_ensemble
    else:
      model_config.data.eval.num_ensemble = num_ensemble
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
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    search_and_predict(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        data_pipeline=data_pipeline,
        model_runners=model_runners,
        amber_relaxer=amber_relaxer,
        benchmark=FLAGS.benchmark,
        clear_cache=FLAGS.clear_cache,
        models_to_relax=FLAGS.models_to_relax,
        use_precomputed_features=FLAGS.use_precomputed_features,
        model_type=model_type,
    )


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
      'uniref90_database_path',
      'mgnify_database_path',
      'template_mmcif_dir',
      'max_template_date',
      'obsolete_pdbs_path',
      'use_gpu_relax',
  ])

  app.run(main)
