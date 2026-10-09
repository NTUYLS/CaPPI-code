"""Write the Chinese revision readout only after 50+40 runs and 41 tests pass.

Reads independently verified result exports. No training, model selection,
statistical significance testing, or manuscript-contract changes occur here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
NEURAL_LABELS = {
    "pair__random__drw__residual": "主模型：残差融合 / DRW",
    "pair__random__drw__concat_matched": "容量匹配拼接 MLP",
    "pair__random__drw__concat_linear": "线性拼接融合",
    "pair__random__drw__sequence_only": "仅序列",
    "pair__random__drw__topology_only": "仅拓扑",
    "pair__random__deepwalk__residual": "均匀游走 / DeepWalk-style，共享 SGNS",
    "pair__random__node2vec__residual": "node2vec 游走，共享 SGNS",
    "pair__degree_matched__drw__residual": "主模型 / 近似度匹配负例",
    "protein__random__drw__residual": "两端未知 / 主模型 / 零拓扑",
    "protein__random__drw__sequence_only": "两端未知 / 仅序列",
}
CLASSICAL_LABELS = {
    "degree_product": "训练图端点 degree 乘积",
    "common_neighbor": "训练图共同邻居数",
    "aa20_drw_lr": "AA20 + DRW / 逻辑回归",
    "aa20_drw_rf": "AA20 + DRW / 随机森林",
}
FULL = "pair__random__drw__residual"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"尚未产生必需的验证产物：{path}；不生成最终结论。")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def checked_index(rows: list[dict], expected: set[str]) -> dict[str, dict]:
    keys = [row["experiment"] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("结果条件列表不完整、重复或含有未经约定的条件")
    result = {row["experiment"]: row for row in rows}
    for row in rows:
        if int(row["completed_folds"]) != 5:
            raise ValueError("至少一个结果条件未完成五折")
        for metric in ("auroc", "average_precision"):
            mean, sd = float(row[metric + "_mean"]), float(row[metric + "_sd"])
            if not math.isfinite(mean) or not math.isfinite(sd) or not 0 <= mean <= 1 or sd < 0:
                raise ValueError("无效的五折结果数值")
    return result


def validate_evidence(base: Path) -> tuple[dict, dict, dict]:
    result_dir = base / "results"
    neural_verification = read_json(result_dir / "verification.json")
    if neural_verification.get("status") != "passed" or neural_verification.get("verified_runs") != 50:
        raise RuntimeError("神经实验必须全部 50 项通过逐预测验证，才能生成最终结论。")
    classical_verification = read_json(result_dir / "classical_verification.json")
    if (classical_verification.get("status") != "passed" or classical_verification.get("verified_runs") != 40
            or classical_verification.get("missing_runs") or not classical_verification.get("outputs_written")):
        raise RuntimeError("经典对照必须全部 40 项通过逐预测验证，才能生成最终结论。")

    neural = checked_index(read_csv(result_dir / "verified_summary.csv"), set(NEURAL_LABELS))
    classical_expected = {f"{protocol}__random__{method}" for protocol in ("pair", "protein") for method in CLASSICAL_LABELS}
    classical = checked_index(read_csv(result_dir / "verified_classical_summary.csv"), classical_expected)
    if sha(result_dir / "verified_classical_summary.csv") != classical_verification["verified_summary_sha256"]:
        raise ValueError("经典结果汇总文件与验证记录的哈希不一致")
    campaign = base / "runs" / "reviewer_v1"
    if sha(campaign / "campaign_config.json") != neural_verification["campaign_config_sha256"]:
        raise ValueError("神经实验配置在验证后发生变化")

    # The neural verifier currently stores per-run prediction hashes. Crosscheck
    # them and recompute summary mean/SD from its verified per-fold metric table.
    expected_runs = {f"{key}__fold{fold}" for key in NEURAL_LABELS for fold in range(5)}
    verified_runs = neural_verification.get("runs", [])
    if len(verified_runs) != 50 or {item["run"] for item in verified_runs} != expected_runs:
        raise ValueError("神经验证记录缺少预期运行")
    fold_rows = read_csv(result_dir / "verified_fold_metrics.csv")
    fold_keys = [(row["experiment"], int(row["fold"])) for row in fold_rows]
    if len(fold_keys) != 50 or len(set(fold_keys)) != 50 or set(fold_keys) != {(key, fold) for key in NEURAL_LABELS for fold in range(5)}:
        raise ValueError("神经逐折指标表条件或fold不完整")
    fold_lookup = dict(zip(fold_keys, fold_rows))
    for item in verified_runs:
        folder = campaign / item["run"]
        if sha(folder / "test_predictions.csv") != item["prediction_sha256"]:
            raise ValueError(f"神经预测文件在验证后发生变化：{folder.name}")
        completed = read_json(folder / "completed.json")
        group, fold_text = item["run"].rsplit("__fold", 1)
        metrics = fold_lookup[group, int(fold_text)]
        if completed["name"] != item["run"] or completed["predictions_sha256"] != item["prediction_sha256"]:
            raise ValueError("神经完成记录与验证记录不一致")
        for metric in ("auroc", "average_precision"):
            if not math.isclose(float(metrics[metric]), completed["test_metrics"][metric], rel_tol=0, abs_tol=1e-10):
                raise ValueError("神经逐折指标与已验证完成记录不一致")
    for key, row in neural.items():
        selected = [item for item in fold_rows if item["experiment"] == key]
        for metric in ("auroc", "average_precision"):
            values = [float(item[metric]) for item in selected]
            if (not math.isclose(statistics.mean(values), float(row[metric + "_mean"]), rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(statistics.stdev(values), float(row[metric + "_sd"]), rel_tol=0, abs_tol=1e-10)):
                raise ValueError("神经汇总表与五折指标重新计算的均值/标准差不一致")
    classical_runs = classical_verification.get("runs", [])
    if len(classical_runs) != 40 or {item["run"] for item in classical_runs} != {f"{key}__fold{fold}" for key in classical_expected for fold in range(5)}:
        raise ValueError("经典验证记录缺少预期运行")
    for item in classical_runs:
        folder = base / "runs" / "classical_controls_v1" / item["run"]
        if sha(folder / "completed.json") != item["completed_sha256"] or sha(folder / "test_predictions.csv") != item["prediction_sha256"]:
            raise ValueError("经典运行记录或预测文件在验证后发生变化")

    tests = read_json(base / "test_validation.json")
    if (tests.get("success") is not True or tests.get("tests_run") != 41
            or any(tests.get(key, 0) != 0 for key in ("failures", "errors", "skipped"))):
        raise RuntimeError("需有 41 项全部通过且无跳过的测试记录，才能生成本结论文件。")
    if tests["entrypoint_sha256"] != sha(base / "run_tests.py") or tests["log_sha256"] != sha(base / "test_validation.log"):
        raise ValueError("测试入口或测试日志在验证后发生变化")
    for filename, digest in tests["test_source_sha256"].items():
        if sha(base / "tests" / filename) != digest:
            raise ValueError(f"测试源码在验证后发生变化：{filename}")
    return neural, classical, tests


def format_metric(row: dict, metric: str) -> str:
    return f"{float(row[metric + '_mean']):.4f} ± {float(row[metric + '_sd']):.4f}"


def difference_sentence(neural: dict, comparator: str, label: str) -> str:
    deltas = {metric: float(neural[comparator][metric + "_mean"]) - float(neural[FULL][metric + "_mean"])
              for metric in ("auroc", "average_precision")}
    signs = [1 if delta > 1e-12 else -1 if delta < -1e-12 else 0 for delta in deltas.values()]
    if signs == [1, 1]:
        finding = "两项五折均值均高于主模型；本结果不支持主模型优于该对照。"
    elif signs == [-1, -1]:
        finding = "两项五折均值均低于主模型；这仅是当前固定协议下的描述性差异。"
    elif signs == [0, 0]:
        finding = "两项五折均值均与主模型相同。"
    else:
        finding = "两项指标的方向不一致或存在持平，不能归纳为整体优于或劣于主模型。"
    return f"{label}相对主模型的 AUROC 差为 {deltas['auroc']:+.4f}、AP 差为 {deltas['average_precision']:+.4f}。{finding}"


def render(base: Path, neural: dict, classical: dict, tests: dict) -> str:
    def link(label: str, relative: str) -> str:
        return f"[{label}](<{(base / relative).as_posix()}>)"

    lines = [
        "# CaPPI 返修实验结论",
        "",
        "本文件仅在 50 项神经实验、40 项经典对照全部通过逐预测结果核验，且 41 项契约测试全部通过后生成。下列 10 组神经实验和 8 组经典对照的数值来自修复版的固定数据、划分和训练协议；另行说明论文中单独保留的历史基线结果，二者不合并排名。保留作者指定的 HCC 研究方向和现有输入；这些结果属于计算基准评价，不是 HCC 生物学验证。",
        "",
        "## 如何解释结果",
        "",
        difference_sentence(neural, "pair__random__drw__concat_matched", "容量匹配拼接 MLP"),
        "",
        difference_sentence(neural, "pair__random__drw__topology_only", "仅拓扑模型"),
        "",
        "以上比较限于相同 random-negative、pair 五折的 AUROC 和 AP。参数量匹配不代表函数空间和优化难度完全相同；仅拓扑对照也改变了可用模态与模型容量。未进行显著性检验，不能使用“显著提升”或把微小均值差异当作结构创新的证明。",
        "",
    ]
    matched = neural["pair__degree_matched__drw__residual"]
    random = neural[FULL]
    lines += [f"近似度匹配负例相对随机负例，主模型 AUROC 均值变化 {float(matched['auroc_mean'])-float(random['auroc_mean']):+.4f}、AP 均值变化 {float(matched['average_precision_mean'])-float(random['average_precision_mean']):+.4f}。负池有 26.07% 的样本需要最近箱补足，不能写成完全匹配或已经消除 degree 相关捷径。", "",
              "protein 协议的两个测试端点都未出现在训练图；拓扑明确置零，序列仍可用。它与 pair 协议的测试人群、样本数和正例比例不同，不能把分数差异归因为一个独立因素，也不能将蛋白 ID 分离称为序列同源性 OOD。", "",
              f"两端未知协议下主模型 MCC 为 {format_metric(neural['protein__random__drw__residual'], 'mcc')}。在预先固定的 0.5 阈值下，每折均只预测一个类别；排序指标不能代替有效的二分类表现。稿件已如实披露此结果，未事后调整测试阈值。", "",
              "## 10 组神经实验", "",
              "均值 ± 五折样本标准差；标准差不是置信区间。AP 指 Average Precision，与梯形积分 PR 面积分开。", "",
              "| 模型或控制 | 协议 / 负池 | AUROC | AP |", "|---|---|---|---|"]
    for key, label in NEURAL_LABELS.items():
        protocol, negative = key.split("__")[:2]
        setting = ("两端未知" if protocol == "protein" else "蛋白对") + " / " + ("近似度匹配" if negative == "degree_matched" else "随机")
        lines.append(f"| {label} | {setting} | {format_metric(neural[key], 'auroc')} | {format_metric(neural[key], 'average_precision')} |")
    lines += ["", "## 8 组经典对照", "",
              "均使用相同 random-negative 划分。degree 乘积和共同邻居是未经校准的排名分数，只有 AUROC/AP，分类 Accuracy/MCC 未定义。两端未知协议下这两个图分数均为零，AUROC 为 0.5，AP 等于各 fold 的正例比例；这不代表有效的未知蛋白预测能力。", "",
              "| 对照 | 协议 | AUROC | AP |", "|---|---|---|---|"]
    for protocol in ("pair", "protein"):
        for method, label in CLASSICAL_LABELS.items():
            row = classical[f"{protocol}__random__{method}"]
            setting = "两端未知" if protocol == "protein" else "蛋白对"
            lines.append(f"| {label} | {setting} | {format_metric(row, 'auroc')} | {format_metric(row, 'average_precision')} |")
    lines += ["", "这些是透明定义的 AA20/DRW 特征分类器和图启发式，不是 PASNVGA 或 SDNN-PPI 的复现。它们与主模型的本轮比较只使用相同新协议下的结果。", "",
              "## 单独保留的历史基线", "",
              "按作者决定，本轮不再重跑 PASNVGA 和 SDNN-PPI。两种方法在原稿中报告的全部 6 项分类指标及 AUC/AUPR 保留在论文独立的历史结果 Table 5A/B，保留原数值和指标名称；旧 CaPPI 结果不回填。历史 AUPR 不改称本轮 Average Precision。", "",
              "保留历史结果不属于本轮 50 项神经实验和 40 项经典对照，也不代表这些旧基线已经按新划分、负例和训练协议重新运行。历史表与本轮主模型及新增控制分开解释，不混排、不计算跨协议提升，也不据此宣称新 CaPPI 优于这两种已发表方法。两项旧基线重跑已从当前待办移除；协议差异造成的可比性限制继续保留。", "",
              "## 已提供的返修证据", "",
              "| 审稿问题 | 本轮已经提供的证据 |", "|---|---|",
              "| R2.1、R2.6：融合贡献及容量匹配 | 同划分的残差融合、参数差 89 的拼接 MLP、线性拼接和单模态对照；结论按实际均值差异限定。 |",
              "| R2.2、R2.3：蛋白重叠与图信息泄漏 | 真正互斥的五折测试集合；各 fold 只用训练阳性边建图；增加两端未知蛋白测试及明确零拓扑策略。 |",
              "| R2.5：负例构造敏感性 | 相同主模型与训练规则下的随机、近似度匹配负例对照，披露匹配不足及推定负例性质。 |",
              "| R2.6：DRW 对照 | DRW、均匀游走/DeepWalk-style、node2vec 共享训练图和 SGNS；游走统计仅解释路径行为。 |",
              "| R2.6：已发表方法基线 | PASNVGA、SDNN-PPI 的原稿 6 项分类指标及 AUC/AUPR 单独保留为历史 Table 5A/B；不回填旧 CaPPI，不与新协议结果混排，不声称已重跑。 |",
              "| R3.1：训练过程不清 | GCNN、投影、融合与分类器联合训练；训练图 Skip-gram 固定；保存配置、参数、梯度、checkpoint 和预测记录。 |",
              "| R3.2：选择程序不清 | 配置预先确定，无本轮超参数搜索；验证 BCE 选择 checkpoint，固定测试阈值 0.5；未称“最优配置”。 |",
              "| 修改可审查性 | 新代码、数据指纹、逐样本预测、独立结果核验与测试记录；稿件修改仍需逐条对应实际证据。 |", "",
              "以上为已完成的实现与实验响应，不等于编辑或审稿人已认可创新性或生物学结论。", "",
              "## 仍未完成的证据与投稿工作", "",
              "- 独立外部数据集评价，以及更严格的序列相似性分离评价。",
              "- 匹配的一般/非 HCC 网络对照或跨域实验；尚不能认定模型学习到了 HCC 特异机制。",
              "- 根据原始下载和筛选记录补全蛋白选择依据、原始 STRING 标识映射、版本/日期、阈值和证据通道。程序未重新推断这些来源信息。",
              "- 按编辑要求将确切修订版代码存入分配 DOI 的仓储，并在 Methods 或 Code Availability 给出版本及 DOI。当前本地文件不构成 DOI 发布。",
              "- 完成人工核对、标记稿和逐条回复的最终审阅；独立疾病或实验验证缺失时，不能保留超出证据的疾病机制或临床应用结论。", "",
              "## 41 项测试及可追溯文件", "",
              f"测试记录显示：{tests['tests_run']} 项通过，0 失败、0 错误、0 跳过。测试时间为 {tests.get('started_at', '见测试记录')}。这些是合成数据与实现契约检查，不代替正式训练结果或生物学验证。", "",
              f"证据：{link('测试JSON', 'test_validation.json')}；{link('完整测试日志', 'test_validation.log')}；{link('正确测试入口 run_tests.py', 'run_tests.py')}。", "",
              "本机 Windows 环境使用先导入 PyTorch、再发现测试的入口，避免直接 unittest discovery 的依赖导入顺序冲突。复核命令：", "", "```powershell",
              f"& 'D:\\Anaconda\\envs\\arg_gpu\\python.exe' '{(base / 'run_tests.py').as_posix()}' --cpu-only --threads 2 --expected-tests 41",
              "```", "",
              f"神经证据：{link('50项验证记录', 'results/verification.json')}；{link('神经汇总CSV', 'results/verified_summary.csv')}；{link('逐折指标CSV', 'results/verified_fold_metrics.csv')}。", "",
              f"经典对照证据：{link('40项验证记录', 'results/classical_verification.json')}；{link('经典汇总CSV', 'results/verified_classical_summary.csv')}。", "",
              f"协议与复现：{link('原实验契约说明', 'README_返修实验说明.md')}；{link('数据manifest', 'prepared/manifest.json')}；{link('游走诊断', 'results/walk_diagnostics.json')}。本结果文件补充契约中的实施状态，不改写已经固定的实验规则。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "RESULTS_返修结论.md")
    parser.add_argument("--check-only", action="store_true", help="Validate prerequisites without writing the final result document")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        neural, classical, tests = validate_evidence(args.base.resolve())
        if args.check_only:
            print("全部先决证据通过；未写结果文档。")
        else:
            text = render(args.base.resolve(), neural, classical, tests)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".md.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(args.output)
            print(str(args.output.resolve()))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"未生成最终返修结论：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
