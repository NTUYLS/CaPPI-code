"""Summarize frozen missing-topology runs without tuning or selecting models.

The independent verifier must pass the complete selected-test campaign before
RESULTS.md can be written. Seed summaries first average the five held-out folds;
campaign means/SDs then operate on the three model-training repeats, not fifteen
independent observations. All comparisons reuse an adaptively examined benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch  # Initialize Windows Torch DLLs before importing sklearn/scipy.
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score,
)


HERE = Path(__file__).resolve().parent
METRICS = ("accuracy", "sensitivity", "specificity", "precision", "f1", "mcc",
           "auroc", "aupr", "average_precision")
REPAIR_FAMILIES = ("topology_dropout", "topology_dropout_bypass")
COMPARATORS = ("residual", "topology_only", "sequence_only", "concat_matched")
LABELS = {
    "residual": "原完整融合", "topology_dropout": "拓扑随机屏蔽",
    "topology_dropout_bypass": "拓扑屏蔽＋序列旁路", "sequence_only": "仅序列",
    "topology_only": "仅拓扑", "concat_matched": "容量匹配 MLP",
}
BENCHMARK_SCOPE = "adaptive_same_benchmark_no_fresh_independent_test"
BASE_SEED = 20260930


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    require(path.is_file(), f"Required JSON missing: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_path(value, root_run: Path) -> Path:
    if isinstance(value, dict):
        value = value.get("out", value.get("run_dir", value.get("path")))
    require(isinstance(value, str) and bool(value), "Selected run requires out/run_dir/path")
    path = Path(value)
    path = (root_run / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(root_run.resolve()), f"Selected run lies outside campaign: {path}")
    return path


def selected_paths(index: dict, root_run: Path) -> list[Path]:
    values = index.get("selected_runs", index.get("runs"))
    require(isinstance(values, list) and bool(values), "selected_index requires a nonempty selected_runs/runs list")
    paths = [run_path(value, root_run) for value in values]
    require(len(set(paths)) == len(paths), "Duplicate selected run directory")
    return paths


def prediction_metrics(path: Path) -> tuple[dict, dict]:
    frame = pd.read_csv(path)
    required = {"protein_u", "protein_v", "label", "probability", "prediction"}
    require(required.issubset(frame), f"Prediction columns missing: {path}")
    labels, probability = frame.label.to_numpy(), frame.probability.to_numpy(dtype=float)
    require(len(frame) > 0 and set(labels.tolist()) == {0, 1}, f"Both binary label classes required: {path}")
    require(np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1)),
            f"Invalid prediction probabilities: {path}")
    predicted = (probability >= .5).astype(np.int64)
    require(np.array_equal(predicted, frame.prediction.to_numpy()), f"Prediction threshold differs: {path}")
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, probability)
    metrics = {
        "n": int(len(labels)), "positive_n": int(labels.sum()), "threshold": .5,
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "aupr": float(auc(recall, precision)),
    }
    pair_rows = frame[["protein_u", "protein_v", "label"]].to_numpy(dtype=np.int64)
    extra = {
        "predicted_positive_n": int(predicted.sum()), "predicted_negative_n": int((predicted == 0).sum()),
        "probability_min": float(probability.min()), "probability_max": float(probability.max()),
        "constant_score": bool(float(np.ptp(probability)) <= 1e-12),
        "single_class_prediction": bool(len(set(predicted.tolist())) == 1),
        "positive_only_prediction": bool(np.all(predicted == 1)),
        "negative_only_prediction": bool(np.all(predicted == 0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pair_fingerprint": hashlib.sha256(np.ascontiguousarray(pair_rows).tobytes()).hexdigest(),
    }
    return metrics, extra


def check_saved_metrics(calculated: dict, saved: dict, context: str) -> None:
    for name, value in calculated.items():
        saved_name = "pr_trapezoid_area" if name == "aupr" else name
        require(saved_name in saved and isinstance(saved[saved_name], (int, float))
                and np.isfinite(saved[saved_name])
                and np.isclose(saved[saved_name], value, atol=1e-10, rtol=0),
                f"Saved metric differs: {context}/{saved_name}")


def expected_keys(contract: dict) -> set[tuple]:
    return {
        (condition["protocol"], condition["negative"], family, int(fold), int(repeat))
        for condition in contract["conditions"] for family in condition["families"]
        for fold in contract["folds"] for repeat in contract["repeat_indices"]
    }


def verification_rows(path: Path | None, paths: list[Path], index_path: Path) -> tuple[dict | None, dict]:
    if path is None:
        return None, {}
    report = read_json(path)
    require(report.get("status") == "passed" and report.get("scope") == "selected_test",
            "Independent selected-test verification must pass before a final report is written")
    require(report.get("verified_runs") == len(paths), "Verifier does not cover every selected test run")
    rows = report.get("runs")
    require(isinstance(rows, list) and len(rows) == len(paths), "Verifier run-list length differs")
    verified = {Path(row["run_dir"]).resolve(): row for row in rows}
    require(len(verified) == len(rows) and set(verified) == set(paths), "Verifier run paths differ from frozen selection")
    if "selected_index_sha256" in report:
        require(report["selected_index_sha256"] == sha(index_path), "Verifier belongs to another selection index")
    for row in rows:
        require(row.get("status") == "test_evaluated", "Validation-only run cannot be summarized as a test")
    return report, verified


def extract_run(folder: Path, contract: dict, verified: dict | None, verification_status: str) -> dict:
    complete = read_json(folder / "completed.json")
    test = read_json(folder / "test_completed.json")
    declaration = read_json(folder / "selection_frozen.json")
    config = complete["config"]
    family = config["settings"]["family"]
    fold, model_seed, graph_seed = int(config["fold"]), int(config["model_seed"]), int(config["graph_seed"])
    offset = model_seed - BASE_SEED - fold
    require(offset >= 0 and offset % 1000 == 0, "Model seed cannot be assigned to the fixed repeat formula")
    repeat = offset // 1000
    require(repeat in contract["repeat_indices"], "Model seed repeat is outside the campaign")
    require(graph_seed == BASE_SEED + fold, "Graph seed is not fixed across model repeats")
    require(complete.get("test_evaluated") is False and test.get("test_evaluated") is True
            and test.get("evaluation_partition") == "test", "Training and test phases are not correctly separated")
    require(declaration.get("selected_before_test") is True and bool(declaration.get("selection_rule")),
            "Selected checkpoint was not frozen before test evaluation")
    if "repeat" in declaration:
        require(int(declaration["repeat"]) == repeat, "Frozen repeat differs from the model seed")
    if "learning_rate" in declaration:
        require(float(declaration["learning_rate"]) == float(config["learning_rate"]),
                "Frozen learning rate differs from training configuration")
    for field in ("run_signature", "checkpoint_sha256", "split_hash"):
        require(test[field] == complete[field], f"Training/test provenance differs: {folder}/{field}")
    for field in ("run_signature", "checkpoint_sha256"):
        require(declaration[field] == complete[field], f"Frozen selection differs: {folder}/{field}")
    require(test["selection_declaration_sha256"] == sha(folder / "selection_frozen.json"),
            "Frozen selection changed after test evaluation")
    validation_path, test_path = folder / "validation_predictions.csv", folder / "test_predictions.csv"
    validation_sha, test_sha = sha(validation_path), sha(test_path)
    require(complete["artifact_sha256"]["validation_predictions.csv"] == validation_sha,
            "Validation prediction content changed")
    require(test["artifact_sha256"]["test_predictions.csv"] == test_sha, "Test prediction content changed")
    if verified is not None:
        require(verified["checkpoint_sha256"] == test["checkpoint_sha256"]
                and verified["run_signature"] == test["run_signature"]
                and verified["split_hash"] == test["split_hash"], "Independent verification identity differs")
        require(verified["validation_prediction_sha256"] == validation_sha
                and verified["test_prediction_sha256"] == test_sha
                and verified["selection_freeze_sha256"] == sha(folder / "selection_frozen.json"),
                "Artifacts changed after independent verification")
    validation_metrics, validation_extra = prediction_metrics(validation_path)
    test_metrics, test_extra = prediction_metrics(test_path)
    check_saved_metrics(validation_metrics, complete["validation_metrics"], str(folder / "completed.json"))
    check_saved_metrics(test_metrics, test["test_metrics"], str(folder / "test_completed.json"))
    if verified is not None:
        check_saved_metrics(validation_metrics, verified["validation_metrics"], "independent validation metrics")
        check_saved_metrics(test_metrics, verified["test_metrics"], "independent test metrics")
    row = {
        "protocol": config["protocol"], "negative": config["negative_sampling"], "family": family,
        "fold": fold, "repeat": repeat, "model_seed": model_seed, "graph_seed": graph_seed,
        "learning_rate": float(config["learning_rate"]), "best_epoch": int(complete["best_epoch"]),
        "epochs_completed": int(complete["epochs_completed"]),
        "best_validation_bce": float(complete["best_validation_bce"]),
        "run_dir": str(folder), "run_signature": test["run_signature"],
        "checkpoint_sha256": test["checkpoint_sha256"], "split_hash": test["split_hash"],
        "graph_hash": test["graph_hash"], "validation_prediction_sha256": validation_sha,
        "test_prediction_sha256": test_sha,
        "selection_freeze_sha256": test["selection_declaration_sha256"],
        "evaluation_scope": BENCHMARK_SCOPE, "fresh_independent_test": False,
        "verification_status": verification_status,
    }
    row.update({"validation_" + key: value for key, value in {**validation_metrics, **validation_extra}.items()})
    row.update({"test_" + key: value for key, value in {**test_metrics, **test_extra}.items()})
    return row


def summarize_repeats(detail: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows, summary_rows = [], []
    numeric_metrics = [partition + "_" + metric for partition in ("validation", "test") for metric in METRICS]
    numeric_metrics.append("best_validation_bce")
    keys = ["protocol", "negative", "family"]
    for identity, group in detail.groupby(keys, sort=True):
        group_seeds = []
        for repeat, seed_group in group.groupby("repeat", sort=True):
            require(set(seed_group.fold) == set(contract["folds"]) and len(seed_group) == len(contract["folds"]),
                    f"Repeat does not contain exactly the planned folds: {identity}/{repeat}")
            row = dict(zip(keys, identity))
            row.update({"repeat": int(repeat), "n_folds": len(seed_group),
                        "model_seeds": ";".join(map(str, seed_group.sort_values("fold").model_seed)),
                        "evaluation_scope": BENCHMARK_SCOPE, "fresh_independent_test": False,
                        "verification_status": str(seed_group.verification_status.iloc[0]),
                        "test_single_class_folds": int(seed_group.test_single_class_prediction.sum()),
                        "test_positive_only_folds": int(seed_group.test_positive_only_prediction.sum()),
                        "test_negative_only_folds": int(seed_group.test_negative_only_prediction.sum()),
                        "test_constant_score_folds": int(seed_group.test_constant_score.sum()),
                        "test_mcc_zero_folds": int(np.isclose(seed_group.test_mcc, 0, atol=1e-12, rtol=0).sum())})
            for metric in numeric_metrics:
                row[metric + "_mean"] = float(seed_group[metric].mean())
                row[metric + "_fold_sd"] = float(seed_group[metric].std(ddof=1))
            seed_rows.append(row)
            group_seeds.append(row)
        require({row["repeat"] for row in group_seeds} == set(contract["repeat_indices"]), "Training repeats missing")
        row = dict(zip(keys, identity))
        row.update({"n_model_repeats": len(group_seeds), "n_folds_per_repeat": len(contract["folds"]),
                    "n_selected_runs": len(group), "evaluation_scope": BENCHMARK_SCOPE,
                    "fresh_independent_test": False,
                    "verification_status": str(group.verification_status.iloc[0]),
                    "sd_definition": "sample SD across model repeats after each repeat averages all folds",
                    "fold_sd_definition": "mean of within-repeat sample SDs across dependent folds",
                    "test_single_class_runs": int(group.test_single_class_prediction.sum()),
                    "test_positive_only_runs": int(group.test_positive_only_prediction.sum()),
                    "test_negative_only_runs": int(group.test_negative_only_prediction.sum()),
                    "test_constant_score_runs": int(group.test_constant_score.sum()),
                    "test_mcc_zero_runs": int(np.isclose(group.test_mcc, 0, atol=1e-12, rtol=0).sum())})
        for metric in numeric_metrics:
            values = np.asarray([seed[metric + "_mean"] for seed in group_seeds], dtype=float)
            fold_sds = np.asarray([seed[metric + "_fold_sd"] for seed in group_seeds], dtype=float)
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_seed_sd"] = float(values.std(ddof=1))
            row[metric + "_fold_sd_mean"] = float(fold_sds.mean())
        summary_rows.append(row)
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def paired_differences(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (protocol, negative, fold, repeat), group in detail.groupby(["protocol", "negative", "fold", "repeat"], sort=True):
        lookup = {row.family: row for row in group.itertuples(index=False)}
        for repair in REPAIR_FAMILIES:
            if repair not in lookup:
                continue
            for comparator in COMPARATORS:
                if comparator not in lookup:
                    continue
                a, b = lookup[repair], lookup[comparator]
                require(a.split_hash == b.split_hash and a.graph_seed == b.graph_seed
                        and a.graph_hash == b.graph_hash and a.model_seed == b.model_seed
                        and a.test_pair_fingerprint == b.test_pair_fingerprint,
                        "Paired comparison differs in test membership, graph, or training repeat")
                row = {"protocol": protocol, "negative": negative, "fold": int(fold), "repeat": int(repeat),
                       "repair": repair, "comparator": comparator, "difference_direction": "repair_minus_comparator",
                       "model_seed": int(a.model_seed), "graph_seed": int(a.graph_seed), "split_hash": a.split_hash,
                       "evaluation_scope": BENCHMARK_SCOPE, "fresh_independent_test": False,
                       "verification_status": a.verification_status,
                       "statistical_scope": "descriptive paired differences; no p-values"}
                for metric in METRICS:
                    row["test_" + metric + "_delta"] = float(getattr(a, "test_" + metric) - getattr(b, "test_" + metric))
                rows.append(row)
    return pd.DataFrame(rows)


def make_report(detail: pd.DataFrame, seeds: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame,
                verification_report: Path) -> str:
    lines = [
        "# CaPPI 缺失拓扑适配与模态贡献实验", "",
        f"全部 {len(detail)} 个预先冻结的测试评估已完成并通过独立数值及来源核验。核验文件：`{verification_report}`。", "",
        "本轮修改受此前查看的同一基准结果启发，属于适应性复用基准比较；没有新增盲测或独立外部测试集。", "",
        "各模型使用相同划分、负样本和固定训练图；学习率只由第 0 次重复的验证 BCE 选择，后续重复保持该学习率。"
        "检查点取最小验证 BCE，测试分类阈值固定为 0.5。", "",
        "图随机种子及图嵌入在同一折的全部模型和重复之间固定；3 次种子重复反映神经网络训练随机性，"
        "不包含 3 次独立图表示重生成。", "",
        "先在每次模型种子重复内平均 5 折，再对 3 次重复的均值计算均值与样本 SD。"
        "折间 SD 单独保存；训练集重叠的 15 次评估不视为独立样本，未计算显著性 p 值。", "",
        "AUPR 为 PR 曲线梯形积分；Average Precision 单独保存在 CSV 中，两者没有混用。", "",
        "protein 划分保证两个端点的蛋白 ID 均与训练分区不重叠，并排除跨分区蛋白对；"
        "这不等同于序列同源性独立的泛化评估。仅拓扑模型在该划分的验证/测试端点上没有可用拓扑，"
        "其常数预测是预期的可用性对照。", "",
        "本轮保留既有负样本池。度匹配负样本的基准构造使用完整已知正边网络的度信息，"
        "包含留出正边；这些信息不进入训练图、特征或拟合的标准化参数。"
        "负样本仍是未观测的候选非相互作用，并非实验确认的非相互作用。"
        "完整正边集合也用于数据完整性审查；模型拟合及参数选择不读取测试标签数组或测试分数。", "",
        "三个融合模型分别检验原路径、加入拓扑屏蔽，以及在屏蔽基础上增加序列旁路；"
        "没有单独的无屏蔽旁路模型，不能估计旁路的独立主效应。"
        "容量匹配 MLP 未加拓扑屏蔽，修复模型与它的差异同时包含训练方式差异，"
        "不能仅据此证明融合机制优越。", "",
        "## 测试结果", "",
        "| 划分 / 负样本 | 模型 | AUC（均值 ± 种子 SD） | AUPR（均值 ± 种子 SD） | MCC（均值 ± 种子 SD） | 单类预测 / 全部运行 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.protocol} / {row.negative} | {LABELS.get(row.family, row.family)} | "
                     f"{row.test_auroc_mean:.4f} ± {row.test_auroc_seed_sd:.4f} | "
                     f"{row.test_aupr_mean:.4f} ± {row.test_aupr_seed_sd:.4f} | "
                     f"{row.test_mcc_mean:.4f} ± {row.test_mcc_seed_sd:.4f} | "
                     f"{row.test_single_class_runs} / {row.n_selected_runs} |")
    lines += ["", "## 修复实际效果", ""]
    residual_pairs = paired[paired.comparator == "residual"] if len(paired) else pd.DataFrame()
    groups = residual_pairs.groupby(["protocol", "negative", "repair"], sort=True) if len(residual_pairs) else []
    for (protocol, negative, repair), group in groups:
        deltas = group.groupby("repeat")[["test_auroc_delta", "test_aupr_delta", "test_mcc_delta"]].mean()
        mean = deltas.mean()
        auc_values = deltas.test_auroc_delta.to_numpy()
        if np.all(auc_values > 0):
            judgement = "3 次重复的折均值 AUC 均高于原完整融合"
        elif np.all(auc_values <= 0):
            judgement = "3 次重复的折均值 AUC 均未超过原完整融合"
        else:
            judgement = "AUC 的改善方向随模型重复变化，未呈现一致增益"
        lines.append(f"- {protocol} / {negative}，{LABELS[repair]}：{judgement}。"
                     f"平均差值 ΔAUC={mean.test_auroc_delta:+.4f}，ΔAUPR={mean.test_aupr_delta:+.4f}，"
                     f"ΔMCC={mean.test_mcc_delta:+.4f}；这些是同一基准上的描述性差异。")
    lines += ["", "## 与单模态模型的直接比较", ""]
    for protocol, comparator in (("protein", "sequence_only"), ("pair", "topology_only")):
        controls = paired[(paired.protocol == protocol) & (paired.negative == "random")
                          & (paired.comparator == comparator)] if len(paired) else pd.DataFrame()
        for repair, group in controls.groupby("repair", sort=True) if len(controls) else []:
            deltas = group.groupby("repeat")[["test_auroc_delta", "test_aupr_delta", "test_mcc_delta"]].mean()
            mean = deltas.mean()
            values = deltas.test_auroc_delta.to_numpy()
            if np.all(values > 0):
                judgement = f"3 次重复的折均值 AUC 均高于{LABELS[comparator]}"
            elif np.all(values <= 0):
                judgement = f"3 次重复的折均值 AUC 均未超过{LABELS[comparator]}"
            else:
                judgement = f"相对{LABELS[comparator]}的 AUC 增益方向在重复之间不一致"
            lines.append(f"- {protocol} / random，{LABELS[repair]}：{judgement}。"
                         f"平均差值 ΔAUC={mean.test_auroc_delta:+.4f}，ΔAUPR={mean.test_aupr_delta:+.4f}，"
                         f"ΔMCC={mean.test_mcc_delta:+.4f}；均为描述性配对差异，不能据此宣称统计显著。")
    lines += ["", "## 分类退化检查", ""]
    for row in summary.itertuples(index=False):
        if row.test_mcc_zero_runs or row.test_single_class_runs:
            lines.append(f"- {row.protocol} / {row.negative}，{LABELS.get(row.family, row.family)}："
                         f"MCC 为 0 的运行 {row.test_mcc_zero_runs}/{row.n_selected_runs}；"
                         f"全预测负类 {row.test_negative_only_runs} 次、全预测正类 {row.test_positive_only_runs} 次。")
        if row.test_constant_score_runs:
            lines.append(f"- {row.protocol} / {row.negative}，{LABELS.get(row.family, row.family)}："
                         f"常数分数 {row.test_constant_score_runs}/{row.n_selected_runs} 次（最大与最小概率差不超过 1e-12）。")
    if not any(summary.test_mcc_zero_runs) and not any(summary.test_single_class_runs):
        lines.append("所有选定运行在 0.5 阈值下均预测了两个类别，未出现 MCC 为 0 的运行。")
    lines += ["", "MCC 为 0 与单类预测分别检查；AUC/AUPR 的排序表现不能替代固定阈值下的分类质量。", "",
              "对于常数分数，PR 曲线只有少数阈值点，梯形积分会受约定的 (recall=0, precision=1) 端点影响，"
              "可能高于正例比例；这不是有效区分能力。应同时检查 AUC、分数是否恒定及单独保存的 Average Precision。", "",
              "## 每次重复的五折结果", "",
              "| 划分 / 负样本 | 模型 | 重复 | AUC（五折均值 ± 折 SD） | AUPR（五折均值 ± 折 SD） | MCC（五折均值 ± 折 SD） | 单类预测折数 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in seeds.itertuples(index=False):
        lines.append(f"| {row.protocol} / {row.negative} | {LABELS.get(row.family, row.family)} | {row.repeat} | "
                     f"{row.test_auroc_mean:.4f} ± {row.test_auroc_fold_sd:.4f} | "
                     f"{row.test_aupr_mean:.4f} ± {row.test_aupr_fold_sd:.4f} | "
                     f"{row.test_mcc_mean:.4f} ± {row.test_mcc_fold_sd:.4f} | {row.test_single_class_folds} / {row.n_folds} |")
    lines += ["", "逐运行与来源信息见 `run_metrics.csv`；种子重复结果见 `seed_summary.csv`；"
              "总体结果见 `summary.csv`；相同条件、折和模型重复的差值见 `paired_differences.csv`。", "",
              "这些结果只支持当前任务、负例构造及划分条件下的判断。蛋白对划分的高分不能直接证明未知蛋白泛化；"
              "仅拓扑优势不能解释为序列融合必然无用，容量匹配 MLP 结果已保留，修复无改善时也完整报告。", ""]
    return "\n".join(lines)


def collect_results(root_run: Path, result_root: Path, verification_report: Path | None = None) -> pd.DataFrame:
    """Write traceable summaries; final RESULTS.md requires complete verified tests."""
    root_run, result_root = Path(root_run).resolve(), Path(result_root).resolve()
    require(not result_root.is_relative_to(root_run), "Results must be outside the immutable run campaign")
    require(result_root != HERE and result_root != HERE / "results", "Use a new task-scoped result directory")
    contract_path, index_path = root_run / "campaign_contract.json", root_run / "selected_index.json"
    contract, index = read_json(contract_path), read_json(index_path)
    paths = selected_paths(index, root_run)
    expected = expected_keys(contract)
    require(len(paths) == len(expected) == contract["counts"]["selected_test_evaluations"],
            "Frozen selection is not the complete predeclared campaign")
    report, verified = verification_rows(verification_report, paths, index_path)
    require(report is not None or not (result_root / "RESULTS.md").exists(),
            "Refuse unverified collection into a directory containing a final report")
    status = "passed" if report is not None else "unverified"
    rows = [extract_run(folder, contract, verified.get(folder), status) for folder in paths]
    for entry, row in zip(index.get("selected_runs", index.get("runs")), rows):
        if isinstance(entry, dict):
            for name in ("protocol", "negative", "family", "fold", "repeat", "model_seed", "graph_seed", "learning_rate"):
                if name in entry:
                    require(entry[name] == row[name], f"Selection index metadata differs from selected run: {name}")
    detail = pd.DataFrame(rows).sort_values(["protocol", "negative", "family", "repeat", "fold"]).reset_index(drop=True)
    actual = set(detail[["protocol", "negative", "family", "fold", "repeat"]].itertuples(index=False, name=None))
    require(actual == expected and len(detail) == len(actual), "Missing, duplicate, or unexpected selected condition/fold/repeat")
    seeds, summary = summarize_repeats(detail, contract)
    paired = paired_differences(detail)
    result_root.mkdir(parents=True, exist_ok=True)
    for name, frame in (("run_metrics.csv", detail), ("seed_summary.csv", seeds), ("summary.csv", summary),
                        ("paired_differences.csv", paired)):
        frame.to_csv(result_root / name, index=False, encoding="utf-8-sig")
    manifest = {
        "status": "complete_verified" if report is not None else "metrics_exported_unverified_no_final_report",
        "evaluation_scope": BENCHMARK_SCOPE, "fresh_independent_test": False,
        "selected_test_evaluations": len(detail), "contract_sha256": sha(contract_path),
        "selected_index_sha256": sha(index_path), "collector_sha256": sha(Path(__file__)),
        "verification_report": str(verification_report) if verification_report is not None else None,
        "verification_report_sha256": sha(verification_report) if verification_report is not None else None,
        "summary_definition": "average five folds per model repeat, then mean/sample SD of three repeat means",
        "p_values_computed": False,
        "files": {name: sha(result_root / name) for name in
                  ("run_metrics.csv", "seed_summary.csv", "summary.csv", "paired_differences.csv")},
    }
    if report is not None:
        (result_root / "RESULTS.md").write_text(make_report(detail, seeds, summary, paired, Path(verification_report).resolve()), encoding="utf-8")
        manifest["files"]["RESULTS.md"] = sha(result_root / "RESULTS.md")
    write_json(result_root / "collection_status.json", manifest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, default=HERE / "runs/repair_20261007")
    parser.add_argument("--result-root", type=Path, default=HERE / "results/repair_20261007")
    parser.add_argument("--verification-report", type=Path)
    options = parser.parse_args()
    result = collect_results(options.root_run, options.result_root, options.verification_report)
    print(result[["protocol", "negative", "family", "test_auroc_mean", "test_auroc_seed_sd",
                  "test_aupr_mean", "test_mcc_mean", "test_single_class_runs"]].to_string(index=False))
