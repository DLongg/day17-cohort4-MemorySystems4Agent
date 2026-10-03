from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversation dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Calculate recall fraction (0.0 to 1.0) based on expected keywords."""
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for item in expected if item.lower() in answer_lower)
    return matched / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight response quality score combining recall and structure."""
    rec = recall_points(answer, expected)
    if rec == 0.0:
        return 0.25
    # Boost score if the response has structured bullet formatting and clear detail
    bonus = 0.1 if ("\n-" in answer or "\n•" in answer or "bullet" in answer.lower()) else 0.0
    quality = 0.5 + 0.4 * rec + bonus
    return min(1.0, round(quality, 2))


def run_agent_benchmark(
    agent_name: str,
    agent: Any,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over multiple conversations and recall questions.

    Procedure:
    1. Feed all dialogue turns to the agent sequentially.
    2. Ask recall questions in fresh, separate threads to test cross-session memory.
    3. Measure token usage, prompt context tokens, recall accuracy, and memory growth.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    last_user_id = "default_user"

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        last_user_id = user_id
        turns: list[str] = conv.get("turns", [])
        recall_questions: list[dict[str, Any]] = conv.get("recall_questions", [])

        # Dialogue thread
        thread_id = f"thread_{conv_id}"
        for turn in turns:
            agent.reply(user_id, thread_id, turn)

        total_agent_tokens += agent.token_usage(thread_id)
        total_prompt_tokens += agent.prompt_token_usage(thread_id)
        total_compactions += agent.compaction_count(thread_id)

        # Cross-session recall questions asked in fresh threads
        for idx, rq in enumerate(recall_questions):
            recall_thread_id = f"recall_{conv_id}_{idx}"
            question = rq["question"]
            expected = rq.get("expected_contains", [])

            resp = agent.reply(user_id, recall_thread_id, question)
            ans = resp.get("content", "")

            rec = recall_points(ans, expected)
            qual = heuristic_quality(ans, expected)
            recall_scores.append(rec)
            quality_scores.append(qual)

            total_agent_tokens += agent.token_usage(recall_thread_id)
            total_prompt_tokens += agent.prompt_token_usage(recall_thread_id)
            total_compactions += agent.compaction_count(recall_thread_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Memory growth on disk
    memory_growth = (
        agent.memory_file_size(last_user_id)
        if hasattr(agent, "memory_file_size")
        else 0
    )

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 4),
        response_quality=round(avg_quality, 4),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a clean markdown table."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = []
    for r in rows:
        data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality:.2f}",
            f"{r.memory_growth_bytes:,}",
            r.compactions,
        ])
    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github")
    except ImportError:
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in data]
        return "\n".join([header_line, sep_line] + body_lines)


def main() -> None:
    """Run both standard and long-context stress benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    std_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    print("================================================================================")
    print("PHASE 2 - TRACK 3 - DAY 17: MEMORY SYSTEMS BENCHMARK")
    print("================================================================================\n")

    # 1. Standard Benchmark
    print("### Standard Benchmark (data/conversations.json)")
    print("Evaluating 10 conversations with cross-session recall questions...\n")
    std_convs = load_conversations(std_data_path)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    baseline_std_row = run_agent_benchmark("Baseline", baseline_std, std_convs, config)

    advanced_std = AdvancedAgent(config=config, force_offline=True)
    advanced_std_row = run_agent_benchmark("Advanced", advanced_std, std_convs, config)

    print(format_rows([baseline_std_row, advanced_std_row]))
    print("\n" + "-" * 80 + "\n")

    # 2. Long-Context Stress Benchmark
    print("### Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print("Evaluating 16 long turns to stress test compact memory...\n")
    stress_convs = load_conversations(stress_data_path)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    baseline_stress_row = run_agent_benchmark("Baseline", baseline_stress, stress_convs, config)

    advanced_stress = AdvancedAgent(config=config, force_offline=True)
    advanced_stress_row = run_agent_benchmark("Advanced", advanced_stress, stress_convs, config)

    print(format_rows([baseline_stress_row, advanced_stress_row]))
    print("\n================================================================================")


if __name__ == "__main__":
    main()
