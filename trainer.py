import torch
import torch.nn.functional as F

import os, time
import wandb
import pandas as pd
import numpy as np
from dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from utils import calc_dev_risk, calculate_risk #, _plot_umap

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter



class cross_domain_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.num_sweeps = args.num_sweeps

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        self.is_sweep = args.is_sweep
        self.sweep_project = args.sweep_project

        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': "random",
            'metric': {'name': f'dev_risk', 'goal': 'minimize'},
            'name': f"{self.run_description}",
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project, entity='emadeldeen')

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)  # Training with sweep


    def train(self):
        if self.is_sweep:
            wandb.init(config=self.default_hparams)
            run_name = f"sweep_{self.run_description}"
        else:
            run_name = f"{self.run_description}"
            wandb.init(config=self.default_hparams, mode="offline", name=run_name)  # change to 'online' to see on wandb website

        self.hparams = wandb.config
        # Logging
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files.

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.

        self.metrics = {'accuracy': [], 'f1_score': [], 'src_risk': [],
                        'trg_risk': [], 'dev_risk': []}

        start_time = time.time()
        for i in scenarios:
            src_id = i[0]
            trg_id = i[1]

            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)

                # Load data
                self.load_data(src_id, trg_id)

                # get algorithm
                algorithm_class = get_algorithm_class(self.da_method)
                backbone_fe = get_backbone_class(self.backbone)

                algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
                algorithm.to(self.device)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                

                # training..
                for epoch in range(1, self.hparams["num_epochs"] + 1):
                    joint_loaders = enumerate(zip(self.src_train_dl, self.trg_train_dl))
                    algorithm.train()

                    for step, ((src_x, src_y), (trg_x, _)) in joint_loaders:
                        src_x, src_y, trg_x = src_x.float().to(self.device), src_y.long().to(self.device), \
                                              trg_x.float().to(self.device)


                        losses = algorithm.update(src_x, src_y, trg_x)


                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, 32)

                    # logging
                    self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                    for key, val in loss_avg_meters.items():
                        self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                    self.logger.debug(f'-------------------------------------')

                self.algorithm = algorithm
                save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                                self.scenario_log_dir, self.hparams)

                self.evaluate()

                self.calc_results_per_run()

        self.logger.debug(f'Interval = {time.time() - start_time} seconds.')
        # logging metrics
        self.calc_overall_results()
        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics.items()}
        
        wandb.log(average_metrics)
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log({'avg_results': wandb.Table(dataframe=self.averages_results_df, allow_mixed_types=True)})
        wandb.log({'std_results': wandb.Table(dataframe=self.std_results_df, allow_mixed_types=True)})

    def evaluate(self):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss_ = []

        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        with torch.no_grad():
            for data, labels in self.trg_test_dl:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)

                total_loss_.append(loss.item())

                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())

        self.trg_loss = torch.tensor(total_loss_).mean()  # average loss

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs,
                                                             self.hparams)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs,
                                                             self.hparams)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self):
        '''
        Calculates the acc, f1 and risk values for each cross-domain scenario
        '''
        self.acc, self.f1 = _calc_metrics(self.trg_pred_labels, self.trg_true_labels, self.scenario_log_dir,
                                          self.home_path,
                                          self.dataset_configs.class_names)
        self.logger.debug(f"Accuracy: {self.acc} \t F1: {self.f1}")

        if self.is_sweep:
            self.src_risk = calculate_risk(self.algorithm, self.src_test_dl, self.device)
            self.trg_risk = 0 # calculate_risk(self.algorithm, self.trg_test_dl, self.device)
            self.dev_risk = calc_dev_risk(self.algorithm, self.src_train_dl, self.trg_train_dl, self.src_test_dl,
                                        self.dataset_configs, self.device)

            run_metrics = {'accuracy': self.acc,
                           'f1_score': self.f1,
                           'src_risk': self.src_risk,
                           'trg_risk': self.trg_risk,
                           'dev_risk': self.dev_risk}

            df = pd.DataFrame(columns=["acc", "f1", "src_risk", "trg_risk", "dev_risk"])
            df.loc[0] = [self.acc, self.f1, self.src_risk, self.trg_risk,
                         self.dev_risk]
        else:
            run_metrics = {'accuracy': self.acc, 'f1_score': self.f1}
            df = pd.DataFrame(columns=["acc", "f1"])
            df.loc[0] = [self.acc, self.f1]

        for (key, val) in run_metrics.items(): self.metrics[key].append(val)

        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

    def calc_overall_results(self):
        exp = self.exp_log_dir

        # for exp in experiments:
        if self.is_sweep:
            results = pd.DataFrame(
                columns=["scenario", "acc", "f1", "src_risk", "trg_risk", "dev_risk"])
        else:
            results = pd.DataFrame(columns=["scenario", "acc", "f1"])

        scenarios_list = os.listdir(exp)
        scenarios_list = [i for i in scenarios_list if "_to_" in i]
        scenarios_list.sort()


        for scenario in scenarios_list:
            scenario_dir = os.path.join(exp, scenario)
            scores = pd.read_excel(os.path.join(scenario_dir, 'scores.xlsx'))
            scores.insert(0, 'scenario', '_'.join(scenario.split('_')[:-2]))
            results = pd.concat([results, scores])

        avg_results = results.groupby('scenario').mean()
        std_results = results.groupby('scenario').std()

        allover_avg = pd.DataFrame([avg_results.mean()], index=["mean"], columns=avg_results.columns)
        avg_results = pd.concat([avg_results, allover_avg])
        
        allover_std = pd.DataFrame([std_results.mean()], index=["mean"], columns=std_results.columns)
        std_results = pd.concat([std_results, allover_std])

        report_save_path_avg = os.path.join(exp, f"Average_results.xlsx")
        report_save_path_std = os.path.join(exp, f"std_results.xlsx")

        self.averages_results_df = avg_results
        self.std_results_df = std_results
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)
