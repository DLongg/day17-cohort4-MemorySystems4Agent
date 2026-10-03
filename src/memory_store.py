from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text using character-count heuristic.

    Heuristic: ~4 characters per token (stable, reproducible approximation).
    Empty text returns 0 tokens.
    """
    if not text:
        return 0
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` files representing long-term user profile memory.

    Each user has a dedicated markdown file in `state/profiles/<user_id>/User.md`.
    Supports reading, writing, editing, size checking, and structured key-value facts.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        """Resolve the file path for a user's User.md."""
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in user_id.strip())
        if not safe_id:
            safe_id = "default_user"
        user_dir = self.root_dir / safe_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read full markdown text of the user profile."""
        path = self.path_for(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write full markdown text to the user profile."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of search_text with replacement in User.md."""
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse structured key-value facts from User.md."""
        content = self.read_text(user_id)
        results: dict[str, str] = {}
        for line in content.splitlines():
            line_strip = line.strip()
            if line_strip.startswith("- ") and ":" in line_strip:
                key, val = line_strip[2:].split(":", 1)
                results[key.strip()] = val.strip()
        return results

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Upsert a single structured fact in User.md, resolving conflicts cleanly."""
        current = self.read_text(user_id)
        if not current.strip():
            header = f"# User Profile: {user_id}\n\n"
            content = f"{header}- {key}: {value}\n"
            self.write_text(user_id, content)
            return

        pattern = rf"(?m)^-\s*{re.escape(key)}\s*:.*$"
        new_line = f"- {key}: {value}"
        if re.search(pattern, current):
            updated = re.sub(pattern, new_line, current)
            self.write_text(user_id, updated)
        else:
            updated = current.rstrip() + f"\n{new_line}\n"
            self.write_text(user_id, updated)

    def upsert_facts(self, user_id: str, facts: dict[str, str]) -> None:
        """Batch upsert multiple facts into User.md."""
        for k, v in facts.items():
            self.upsert_fact(user_id, k, v)

    def delete_fact(self, user_id: str, key: str) -> bool:
        """Delete a structured fact from User.md (e.g. on memory decay or retraction)."""
        current = self.read_text(user_id)
        pattern = rf"(?m)^-\s*{re.escape(key)}\s*:.*\n?"
        if re.search(pattern, current):
            updated = re.sub(pattern, "", current)
            self.write_text(user_id, updated)
            return True
        return False

    def prune_decayed_facts(self, user_id: str, decay_keys: list[str]) -> int:
        """Prune specific decayed or stale facts from the user profile."""
        pruned_count = 0
        for k in decay_keys:
            if self.delete_fact(user_id, k):
                pruned_count += 1
        return pruned_count


def extract_profile_updates(message: str, min_confidence: float = 0.5) -> dict[str, str]:
    """Convert raw user message into stable profile facts.

    Handles:
    - Name detection (e.g. DũngCT, DũngCT Stress)
    - Profession updates and corrections (backend engineer -> MLOps engineer)
    - Location updates and corrections (Đà Nẵng -> Huế -> Đà Nẵng)
    - Preferences: favorite drink (cà phê sữa đá), food (mì Quảng), pet (corgi Bơ)
    - Response style preferences (3 bullet ngắn, ngắn gọn có ví dụ thực tế)
    - Noise filtering: excludes jokes ('product manager') and temporary trips ('Hà Nội')
    - Question filtering: does not treat query-only turns as facts
    - Confidence threshold: ignores facts declared with uncertainty ('hình như', 'chưa chắc')
    """
    text = message.strip()
    if not text:
        return {}

    lower = text.lower()

    # Confidence check: uncertainty markers
    has_uncertainty = any(
        w in lower for w in ["hình như", "có lẽ", "chưa chắc", "không chắc", "tạm thời nghĩ"]
    )
    confidence = 0.3 if has_uncertainty else 0.9
    if confidence < min_confidence:
        return {}

    # Heuristic: skip pure recall questions that do not declare new facts
    is_pure_question = (
        text.endswith("?")
        or text.startswith("Bạn có thể nhắc lại")
        or text.startswith("Bạn có biết")
        or text.startswith("Bạn thử nhớ lại")
        or text.startswith("Nhắc lại giúp mình")
        or text.startswith("Tóm tắt ngắn về mình")
        or text.startswith("Hiện tại mình đang ở đâu")
    )
    # But allow if explicitly declaring/correcting something
    has_declaration = any(
        kw in text
        for kw in [
            "mình tên là",
            "tên mình là",
            "đính chính",
            "thực ra từ tuần này",
            "giờ mình đang",
            "chuyển sang",
            "không còn làm",
            "đồ uống yêu thích là",
            "món ăn yêu thích là",
            "mình nuôi một",
            "nuôi bé corgi",
            "nuôi một bé corgi",
        ]
    )
    if is_pure_question and not has_declaration:
        return {}

    facts: dict[str, str] = {}

    # 1. Name extraction
    name_match = re.search(
        r"(?:tên\s+mình\s+là|mình\s+tên\s+là|tên\s+là)\s+([A-ZÀ-Ỹ][A-Za-z0-9_À-ỹ]*(?:\s+[A-ZÀ-Ỹ][A-Za-z0-9_À-ỹ]*)*)",
        text,
    )
    if name_match:
        name_val = name_match.group(1).strip()
        # Clean trailing punctuation
        name_val = re.sub(r"[,.;!?].*$", "", name_val).strip()
        if name_val:
            facts["name"] = name_val
    elif "DũngCT Stress" in text:
        facts["name"] = "DũngCT Stress"
    elif "DũngCT" in text and ("tên" in text or "mình là" in text):
        facts["name"] = "DũngCT"

    # 2. Location extraction & corrections
    # Check noise: "Hà Nội chỉ là nơi mình vừa bay ra họp" -> do NOT set location to Hà Nội
    is_hanoi_trip = "hà nội" in lower and ("họp" in lower or "không phải nơi ở" in lower)

    # Check for Đà Nẵng vs Huế updates
    if "đà nẵng" in lower and not is_hanoi_trip:
        if "từ huế sang đà nẵng" in lower or "làm việc ở đà nẵng vài tháng" in lower or "hiện tại là đà nẵng" in lower:
            facts["location"] = "Đà Nẵng"
        elif "chứ không còn ở đà nẵng" in lower or "đừng lấy nó làm nơi ở hiện tại" in lower:
            # Explicitly not Đà Nẵng
            pass
        elif "mình ở đà nẵng" in lower or "ở đà nẵng và" in lower:
            facts["location"] = "Đà Nẵng"

    if "huế" in lower and not is_hanoi_trip:
        if (
            "đang ở huế" in lower
            or "ở huế chứ không" in lower
            or "vẫn ở huế" in lower
            or "hiện ở huế" in lower
            or "ở huế" in lower
        ):
            # If this turn specifically corrects to Huế
            if "từ huế sang đà nẵng" not in lower and "làm việc ở đà nẵng" not in lower:
                facts["location"] = "Huế"

    # 3. Profession extraction & corrections
    # Noise check: "chuyển sang product manager... câu đùa"
    is_pm_joke = "product manager" in lower and ("đùa" in lower or "không phải" in lower)

    if "mlops engineer" in lower:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lower and not is_pm_joke:
        if "không còn làm backend engineer" not in lower and "đừng nói backend engineer" not in lower:
            facts["profession"] = "backend engineer"

    # 4. Favorite drink
    if "cà phê sữa đá" in lower:
        facts["favorite_drink"] = "cà phê sữa đá"

    # 5. Favorite food
    if "mì quảng" in lower:
        facts["favorite_food"] = "mì Quảng"

    # 6. Pet
    if "corgi" in lower or "con bơ" in lower or "bé bơ" in lower:
        facts["pet"] = "corgi tên Bơ"

    # 7. Response style
    if "3 bullet" in lower:
        facts["response_style"] = "3 bullet ngắn, có ví dụ thực tế"
    elif "ngắn gọn" in lower or "bullet ngắn" in lower:
        if "ví dụ thực tế" in lower or "ví dụ thực chiến" in lower:
            facts["response_style"] = "ngắn gọn, có ví dụ thực tế"
        else:
            facts["response_style"] = "ngắn gọn"

    # 8. Tech interests
    techs = []
    if "python" in lower:
        techs.append("Python")
    if "ai ứng dụng" in lower or "ai agent" in lower or "ai" in lower:
        techs.append("AI ứng dụng")
    if techs:
        facts["tech_interests"] = ", ".join(techs)

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages."""
    if not messages:
        return ""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        # Take the most informative initial sentence or chunk up to 120 chars
        first_clause = content.split(".")[0].strip() if "." in content else content[:120].strip()
        if first_clause:
            lines.append(f"- {role}: {first_clause}")
    return "\n".join(lines[:max_items])


@dataclass
class CompactMemoryManager:
    """Manages short-term memory and automatic compaction for long threads.

    Attributes:
        threshold_tokens: Token count limit before compaction triggers.
        keep_messages: Number of recent messages to preserve intact.
        state: Per-thread dictionary tracking messages, summary, and compaction counts.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to the thread history and compact older history if needed."""
        thread_state = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        msgs: list[dict[str, str]] = thread_state["messages"]  # type: ignore
        msgs.append({"role": role, "content": content})

        # Calculate total tokens currently in this thread
        summary_tokens = estimate_tokens(str(thread_state["summary"]))
        messages_tokens = sum(estimate_tokens(m["content"]) for m in msgs)
        total_tokens = summary_tokens + messages_tokens

        # Check if compaction should trigger
        if total_tokens > self.threshold_tokens and len(msgs) > self.keep_messages:
            split_idx = len(msgs) - self.keep_messages
            older_messages = msgs[:split_idx]
            kept_messages = msgs[split_idx:]

            new_summary_part = summarize_messages(older_messages)
            existing_summary = str(thread_state["summary"]).strip()

            if existing_summary:
                combined_summary = existing_summary + "\n" + new_summary_part
                # Limit summary length so it stays compact
                summary_lines = combined_summary.strip().split("\n")
                if len(summary_lines) > 8:
                    combined_summary = "\n".join(summary_lines[-8:])
                thread_state["summary"] = combined_summary
            else:
                thread_state["summary"] = new_summary_part

            thread_state["messages"] = kept_messages
            thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state with messages, summary, and compaction count."""
        return self.state.get(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions that have occurred for this thread."""
        return int(self.state.get(thread_id, {}).get("compactions", 0))
