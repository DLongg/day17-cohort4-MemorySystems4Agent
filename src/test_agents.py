from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated configuration for unit testing."""
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=80,  # Lower threshold to trigger compaction quickly
        compact_keep_messages=2,
        model=ProviderConfig(provider="openai", model_name="mock"),
        judge_model=ProviderConfig(provider="openai", model_name="mock"),
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    profiles_dir = tmp_path / "state" / "profiles"
    store = UserProfileStore(profiles_dir)

    # 1. Write initial profile
    user_id = "test_user_01"
    initial_content = "# User Profile: test_user_01\n\n- name: DũngCT\n- location: Đà Nẵng\n"
    written_path = store.write_text(user_id, initial_content)
    assert written_path.exists()
    assert store.file_size(user_id) > 0

    # 2. Read content
    content = store.read_text(user_id)
    assert "DũngCT" in content
    assert "Đà Nẵng" in content

    # 3. Edit content (location correction)
    success = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert success is True
    updated = store.read_text(user_id)
    assert "Huế" in updated
    assert "Đà Nẵng" not in updated

    # 4. Upsert structured facts
    store.upsert_fact(user_id, "profession", "MLOps engineer")
    facts = store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"
    assert facts.get("location") == "Huế"


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction in AdvancedAgent."""
    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    thread_id = "stress_thread"
    # Send enough long turns to exceed compact_threshold_tokens (80)
    for i in range(8):
        agent.reply(
            "user_stress",
            thread_id,
            f"Turn {i}: Đây là một đoạn hội thoại dài nhằm kiểm tra cơ chế compact memory "
            f"của agent khi vượt qua ngưỡng số token quy định trong cấu hình bài lab.",
        )

    assert agent.compaction_count(thread_id) > 0

    ctx = agent.compact_memory.context(thread_id)
    assert len(ctx["messages"]) <= cfg.compact_keep_messages
    assert ctx["summary"] != ""


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers across sessions while BaselineAgent forgets."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "dungct_recall_test"

    # Thread 1: Introduction with personal facts
    intro_msg = (
        "Chào bạn, mình tên là DũngCT. Mình ở Huế và đang làm MLOps engineer. "
        "Đồ uống yêu thích của mình là cà phê sữa đá."
    )
    baseline.reply(user_id, "thread_session_1", intro_msg)
    advanced.reply(user_id, "thread_session_1", intro_msg)

    # Thread 2: Fresh session asking recall questions
    recall_q = "Mình tên gì, hiện tại làm nghề gì và đồ uống yêu thích là gì?"
    resp_baseline = baseline.reply(user_id, "thread_session_2", recall_q)
    resp_advanced = advanced.reply(user_id, "thread_session_2", recall_q)

    # Baseline must FORGET across fresh threads
    assert "DũngCT" not in resp_baseline["content"]
    assert "MLOps engineer" not in resp_baseline["content"]
    assert "cà phê sữa đá" not in resp_baseline["content"]

    # Advanced must REMEMBER across fresh threads
    assert "DũngCT" in resp_advanced["content"]
    assert "MLOps engineer" in resp_advanced["content"]
    assert "cà phê sữa đá" in resp_advanced["content"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    user_id = "dungct_context_test"
    thread_id = "continuous_long_thread"

    # Feed 10 long turns
    for i in range(10):
        long_message = (
            f"Lượt {i}: Trong kiến trúc bộ nhớ của AI agent, việc xử lý ngữ cảnh dài là một "
            f"thách thức lớn về cả độ trễ và chi phí token. Khi không có compact memory, baseline "
            f"buộc phải gửi lại toàn bộ lịch sử các lượt trước trong mỗi lượt mới, khiến cho prompt "
            f"tokens tăng lên theo hàm bậc hai. Ngược lại, compact memory tóm tắt các lượt cũ và chỉ "
            f"giữ lại một vài tin nhắn gần nhất kèm theo file profile User.md."
        )
        baseline.reply(user_id, thread_id, long_message)
        advanced.reply(user_id, thread_id, long_message)

    prompt_baseline = baseline.prompt_token_usage(thread_id)
    prompt_advanced = advanced.prompt_token_usage(thread_id)

    # Advanced must have significantly reduced prompt tokens compared to Baseline
    assert prompt_advanced < prompt_baseline
    assert advanced.compaction_count(thread_id) > 0


def test_conflict_handling_correction(tmp_path: Path) -> None:
    """Bonus Test: verify that correction properly overrides stale facts in User.md."""
    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    user_id = "dungct_correction"

    # Turn 1: original profession and location
    advanced.reply(user_id, "t1", "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng và làm backend engineer.")
    facts_v1 = advanced.profile_store.facts(user_id)
    assert facts_v1.get("location") == "Đà Nẵng"
    assert facts_v1.get("profession") == "backend engineer"

    # Turn 2: correction
    advanced.reply(user_id, "t1", "À mình đính chính: giờ mình đang ở Huế và chuyển sang MLOps engineer.")
    facts_v2 = advanced.profile_store.facts(user_id)
    # Must hold the new facts without keeping conflicting old facts
    assert facts_v2.get("location") == "Huế"
    assert facts_v2.get("profession") == "MLOps engineer"
    assert "Đà Nẵng" not in advanced.profile_store.read_text(user_id)


def test_confidence_and_noise_filtering(tmp_path: Path) -> None:
    """Bonus Test: verify noise rejection (jokes, trips) and low-confidence filtering."""
    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    user_id = "dungct_noise"

    # Set initial stable fact
    advanced.reply(user_id, "t1", "Mình tên là DũngCT và đang làm MLOps engineer ở Huế.")

    # Send noise message: joke + trip
    advanced.reply(
        user_id,
        "t1",
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại.",
    )
    facts = advanced.profile_store.facts(user_id)
    assert facts.get("profession") == "MLOps engineer"
    assert facts.get("location") == "Huế"
    assert "product manager" not in str(facts)
    assert "Hà Nội" not in str(facts)

    # Low-confidence declaration with hesitation words
    advanced.reply(user_id, "t1", "Hình như có lẽ mình sẽ đổi sở thích sang uống trà xanh.")
    facts_after = advanced.profile_store.facts(user_id)
    assert "trà xanh" not in str(facts_after)


def test_memory_decay_pruning(tmp_path: Path) -> None:
    """Bonus Test: verify pruning stale/decayed facts from User.md."""
    profiles_dir = tmp_path / "state" / "profiles"
    store = UserProfileStore(profiles_dir)
    user_id = "decay_user"

    store.upsert_fact(user_id, "name", "DũngCT")
    store.upsert_fact(user_id, "temp_project", "alpha_hackathon")
    store.upsert_fact(user_id, "location", "Huế")

    assert len(store.facts(user_id)) == 3
    # Prune decayed temporary fact
    pruned = store.prune_decayed_facts(user_id, ["temp_project"])
    assert pruned == 1
    assert "temp_project" not in store.facts(user_id)
    assert store.facts(user_id).get("location") == "Huế"
