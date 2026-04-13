from __future__ import annotations

import copy

from deceptive_fu.config import DATASET_PRESETS, DEFAULT_ARGS
from deceptive_fu.experiment import run_one_experiment
from deceptive_fu.results import save_results_to_csv, summarize_experiment_group, summarize_results


VERIFIER_MODE_LIST = [
    "baseline_drop",
    "cp_only",
    "rep_only",
    "enhanced_joint",
]

FU_PIPELINE_MODE_LIST = [
    "naive_unlearn",
    "finetune_retain",
    "ga_forget_then_repair",
]

SEED_LIST = [42, 43, 44]


def main() -> None:
    all_results = []

    for verifier_mode in VERIFIER_MODE_LIST:
        print("\n" + "=" * 120)
        print(f"Running verifier_mode = {verifier_mode}")
        print("=" * 120)

        for fu_pipeline_mode in FU_PIPELINE_MODE_LIST:
            print("\n" + "-" * 120)
            print(f"Running fu_pipeline_mode = {fu_pipeline_mode}")
            print("-" * 120)

            for dataset_cfg in DATASET_PRESETS:
                print("\n" + "#" * 100)
                print(
                    f"Running dataset = {dataset_cfg['dataset_name']} | "
                    f"verifier = {verifier_mode} | pipeline = {fu_pipeline_mode}"
                )
                print("#" * 100)

                dataset_results = []

                for seed in SEED_LIST:
                    print("\n" + "~" * 80)
                    print(
                        f"Running dataset = {dataset_cfg['dataset_name']}, "
                        f"verifier = {verifier_mode}, "
                        f"pipeline = {fu_pipeline_mode}, "
                        f"seed = {seed}"
                    )
                    print("~" * 80)

                    args_run = copy.deepcopy(DEFAULT_ARGS)
                    args_run.update(dataset_cfg)
                    args_run.update(
                        {
                            "verifier_mode": verifier_mode,
                            "fu_pipeline_mode": fu_pipeline_mode,
                            "backbone": "simple_cnn",
                            "seed": seed,
                            "alpha": 0.5,
                            "attack_variant": "V2",
                            "attack_eps": 1.0,
                            "attack_steps": 10,
                            "attack_lr": 0.1,
                            "cp_beta": 0.05,
                            "cp_resamples": 300,
                            "run_game_analysis": False,
                            "retain_repair_rounds": 5,
                            "retain_repair_lr_scale": 0.5,
                            "ga_steps": 3,
                            "ga_lr": 0.01,
                            "repair_steps_after_ga": 3,
                        }
                    )

                    result = run_one_experiment(args_run)
                    dataset_results.append(result)
                    all_results.append(result)

                    print("\n[Per-run summary]")
                    print(f"dataset = {result['dataset_name']}")
                    print(f"verifier_mode = {result['verifier_mode']}")
                    print(f"fu_pipeline_mode = {result['fu_pipeline_mode']}")
                    print(f"backbone = {result['backbone']}")
                    print(f"seed = {result['seed']}")
                    print(f"Joint Success = {result['joint_success']}")
                    print(f"Verifier Honest Accept = {result['verifier_ref_accept']}")
                    print(f"Verifier Witness Accept = {result['verifier_wit_accept']}")
                    print(f"Serving Acc = {result['serving_acc']:.2f}")
                    print(f"Pipeline Acc = {result['pipeline_acc']:.2f}")
                    print(f"Reference Acc = {result['ref_acc']:.2f}")
                    print(f"Witness Forget Acc = {result['acc_witness_forget']:.2f}")

                summarize_results(
                    dataset_results,
                    tag=(
                        f"Dataset Summary: {dataset_cfg['dataset_name']} | "
                        f"verifier={verifier_mode} | pipeline={fu_pipeline_mode}"
                    ),
                )

    save_results_to_csv(all_results, filepath="results.csv")
    experiment_summary = summarize_experiment_group(all_results)
    save_results_to_csv(experiment_summary, filepath="summary.csv")


if __name__ == "__main__":
    main()
