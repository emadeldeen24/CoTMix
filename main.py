import os
import argparse
import warnings
from trainer import cross_domain_trainer
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='collect_results',          type=str, help='Name of your experiment')
parser.add_argument('--run_description',        default='HHAR_mixup',         type=str, help='name of your runs')
parser.add_argument('--sweep_project',          default='cotmix_hparams_sweep',     type=str, help='name of your sweep project')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='CoTMix',      type=str, help='Our method - but can include any method implemented in algorithms file')

# ========= Select the DATASET ==============
parser.add_argument('--dataset',                default='HHAR',        type=str, help='Dataset of choice: WISDM, EEG, HAR, HHAR')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',         type=str, help='Backbone of choice: CNN')


# ========= Experiment settings ===============
parser.add_argument('--data_path',              default=r'./data/',    type=str,   help='Path containing dataset')
parser.add_argument('--num_runs',               default=3,             type=int,   help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',      type=str,   help='cpu or cuda')
parser.add_argument('--is_sweep',               default=False,         type=bool,  help='singe run or sweep')
parser.add_argument('--num_sweeps',             default=100,           type=str,   help='Number of sweep runs')

args = parser.parse_args()

if __name__ == "__main__":
    trainer = cross_domain_trainer(args)

    if args.is_sweep:
        trainer.sweep()
    else:
        trainer.train()
