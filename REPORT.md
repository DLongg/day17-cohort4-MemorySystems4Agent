# Báo cáo Đánh giá & Phân tích Hệ thống Bộ nhớ cho AI Agent
**Phase 2 — Track 3 — Day 17: Memory Systems for AI Agent**

---

## 1. Giới thiệu & Mục tiêu
Trong bài lab này, chúng ta xây dựng, kiểm thử và đo lường sự khác biệt giữa hai kiến trúc bộ nhớ cho AI Agent:
* **Baseline Agent (Agent A)**: Chỉ có bộ nhớ ngắn hạn trong cùng phiên (`within-session memory`), hoàn toàn quên sạch facts khi chuyển sang thread mới, và kéo theo toàn bộ lịch sử thô khiến chi phí ngữ cảnh phình to.
* **Advanced Agent (Agent B)**: Kiến trúc bộ nhớ đa tầng gồm:
  1. **Short-term Memory**: Lưu trữ tin nhắn trong phiên hiện tại.
  2. **Persistent Memory (`User.md`)**: Lưu trữ bền vững hồ sơ người dùng trên đĩa (`state/profiles/<user_id>/User.md`) qua nhiều phiên làm việc.
  3. **Compact Memory**: Tự động nén ngữ cảnh lịch sử cũ thành tóm tắt và chỉ giữ lại $K$ tin nhắn gần nhất khi tổng số token vượt ngưỡng.

---

## 2. Kết quả Thực nghiệm Benchmark

Hệ thống được đo lường trên cùng bộ dữ liệu kiểm thử tiếng Việt bao gồm hai suite:
1. **Standard Benchmark (`data/conversations.json`)**: 10 cuộc hội thoại thông thường (~10 lượt/cuộc) kèm câu hỏi kiểm tra chéo phiên (`recall_questions` tại thread mới).
2. **Long-Context Stress Benchmark (`data/advanced_long_context.json`)**: 1 hội thoại 16 lượt cực dài chứa nhiều tin tức, quan điểm kỹ thuật và bẫy gây nhiễu để ép bộ nhớ kích hoạt nén liên tục.

### Bảng 1: Standard Benchmark
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 2,295 | 17,162 | **0.0%** | 0.25 | 0 | 0 |
| **Advanced** | 5,336 | 32,722 | **100.0%** | **1.00** | 285 | 3 |

### Bảng 2: Long-Context Stress Benchmark
| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 423 | 22,873 | **0.0%** | 0.25 | 0 | 0 |
| **Advanced** | 1,226 | **9,046** *(giảm 60.5%)* | **100.0%** | **1.00** | 208 | **22** |

---

## 3. Phân tích Chuyên sâu (Theo Bước 8 của Guide & Rubric 75-90 điểm)

### 3.1. Vì sao Advanced Agent có Recall vượt trội hơn Baseline Agent?
* **Baseline Agent** chỉ lưu trữ mảng tin nhắn trong bộ nhớ RAM theo từng `thread_id`. Khi được truy vấn ở một thread mới, trạng thái thread là hoàn toàn mới (empty context). Baseline không thể trả lời bất kỳ câu hỏi nào về tên, nơi ở, nghề nghiệp hay sở thích $\rightarrow$ Recall đạt **0.0%**.
* **Advanced Agent** liên tục bóc tách các fact ổn định qua `extract_profile_updates()` và lưu trữ bền vững vào `User.md`. Khi bắt đầu một thread mới, Agent chủ động nạp `User.md` vào prompt context, cho phép trả lời chính xác 100% các câu hỏi kiểm tra chéo phiên $\rightarrow$ Recall đạt **100.0%**, Response Quality đạt **1.00**.

### 3.2. Vì sao Advanced Agent có thể tiêu tốn nhiều token hơn ở hội thoại ngắn?
* Ở các hội thoại ngắn (~5 đến 10 lượt chat ngắn), Baseline Agent có chi phí rất thấp vì chỉ mang theo vài tin nhắn đơn giản.
* Ngược lại, Advanced Agent chịu một khoản **chi phí cố định (overhead)** ở mỗi lượt:
  1. Phải nạp nội dung file `User.md` vào prompt context.
  2. Tạo câu trả lời chi tiết, có cấu trúc (bullet format) để đảm bảo độ chính xác của profile.
* Do đó, ở Standard Benchmark, Prompt tokens của Advanced cao hơn Baseline (32,722 vs 17,162). Đây là trade-off tất yếu: **Chấp nhận chi phí ngữ cảnh nền để đổi lấy khả năng nhớ xuyên phiên**.

### 3.3. Vì sao Compact Memory giúp Advanced Agent có lợi thế áp đảo ở hội thoại dài?
* Trong kịch bản hội thoại dài (Stress Test 16 lượt dày đặc tin tức và quan điểm), Baseline Agent kéo theo toàn bộ lịch sử thô từ lượt 1 đến lượt 16. Tổng số token ngữ cảnh tích lũy tăng theo cấp số nhân $\sum_{i=1}^N i \sim O(N^2)$, đạt tới **22,873 prompt tokens**.
* **Advanced Agent** thông qua `CompactMemoryManager`:
  * Khi tổng token trong thread vượt ngưỡng quy định (`compact_threshold_tokens = 600`), toàn bộ các tin nhắn cũ được tóm tắt thành một đoạn văn ngắn gọn (`summary`).
  * Chỉ giữ lại $K=4$ tin nhắn gần nhất (`compact_keep_messages`) trong danh sách tin nhắn hoạt động.
  * Ngữ cảnh prompt ở mỗi lượt chỉ bao gồm: `User.md + summary + 4 tin nhắn gần nhất`.
* Nhờ cơ chế này, prompt tokens processed của Advanced chỉ còn **9,046 tokens (tiết kiệm hơn 60.5% lượng ngữ cảnh cần xử lý)** sau **22 lần nén tự động**.

### 3.4. Sự tăng trưởng của File Memory và rủi ro đi kèm trong Production
* Dung lượng `User.md` tăng từ 0 lên 285 bytes sau 10 phiên hội thoại.
* **Các rủi ro kỹ thuật trong môi trường sản xuất**:
  1. **Memory Bloat**: Nếu lưu mọi chi tiết ngẫu nhiên, dung lượng `User.md` sẽ tăng tuyến tính $O(T)$ theo thời gian, khiến chi phí nạp file này vào mỗi lượt chat trở nên quá đắt đỏ.
  2. **Stale Facts & Conflicting Data**: Nếu người dùng thay đổi thông tin (ví dụ: chuyển từ backend sang MLOps, từ Đà Nẵng sang Huế) mà không có cơ chế ghi đè, file memory sẽ chứa các thông tin trái ngược nhau, gây ảo giác (hallucination) cho LLM.
  3. **Noise Pollution**: Người dùng có thể nói đùa, mỉa mai hoặc nhắc đến địa điểm du lịch/công tác tạm thời. Nếu lưu nhầm, hồ sơ người dùng sẽ bị ô nhiễm thông tin sai lệch.

---

## 4. Các Tính năng Bonus Nâng cao (Theo Bước 9 & Rubric 90-100 điểm)

Nhằm giải quyết triệt để các rủi ro trên, hệ thống đã triển khai 4 tính năng mở rộng kỹ thuật:

### 4.1. Conflict Handling & Resolution (Xử lý Đính chính & Xung đột)
* **Vấn đề giải quyết**: Tránh việc lưu đồng thời thông tin cũ và thông tin mới.
* **Cơ chế**: Hàm `UserProfileStore.upsert_fact(user_id, key, value)` sử dụng biểu thức chính quy (regex) để quét và ghi đè dòng `- {key}: {old_value}` thành `- {key}: {new_value}`.
* **Thực nghiệm**: Khi người dùng đính chính: *"Giờ mình ở Huế chứ không còn ở Đà Nẵng"* $\rightarrow$ file `User.md` lập tức thay thế Đà Nẵng bằng Huế, không để lại vết thông tin cũ.

### 4.2. Noise Rejection & Joke Filtering (Lọc Nhiễu & Câu nói đùa)
* **Vấn đề giải quyết**: Ngăn chặn lưu các phát ngôn đùa cợt hoặc địa điểm tạm thời.
* **Cơ chế**:
  * Phát hiện ngữ cảnh đùa: `"product manager... chỉ là câu đùa"` $\rightarrow$ bỏ qua, giữ nguyên nghề nghiệp là MLOps engineer.
  * Phân biệt địa điểm công tác: `"Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày"` $\rightarrow$ không gán nơi ở là Hà Nội.

### 4.3. Confidence Threshold (Ngưỡng Tin cậy)
* **Vấn đề giải quyết**: Tránh lưu các thông tin mà người dùng chưa chắc chắn.
* **Cơ chế**: Tham số `min_confidence` trong `extract_profile_updates`. Khi câu nói chứa các từ thể hiện sự mơ hồ (`"hình như"`, `"có lẽ"`, `"chưa chắc"`), hệ thống gán độ tin cậy thấp (0.3 < 0.5) và từ chối cập nhật vào `User.md`.

### 4.4. Memory Decay & Pruning (Giảm tải & Tỉa gọn Bộ nhớ)
* **Vấn đề giải quyết**: Kiểm soát kích thước file memory đường dài.
* **Cơ chế**: Cung cấp hàm `UserProfileStore.prune_decayed_facts(user_id, decay_keys)` cho phép hệ thống tự động loại bỏ các fact tạm thời (ví dụ: dự án ngắn hạn, hackathon) khi hết hạn.

---

## 5. Kết quả Kiểm thử Toàn diện (`pytest src/test_agents.py -v`)

Hệ thống sở hữu bộ kiểm thử tự động gồm **7 bài test unit**:
```
src/test_agents.py::test_user_markdown_read_write_edit PASSED            [ 14%]
src/test_agents.py::test_compact_trigger PASSED                          [ 28%]
src/test_agents.py::test_cross_session_recall PASSED                     [ 42%]
src/test_agents.py::test_compact_reduces_prompt_load_on_long_thread PASSED [ 57%]
src/test_agents.py::test_conflict_handling_correction PASSED             [ 71%]
src/test_agents.py::test_confidence_and_noise_filtering PASSED           [ 85%]
src/test_agents.py::test_memory_decay_pruning PASSED                     [100%]
============================== 7 passed in 0.15s ==============================
```

---

## 6. Hướng dẫn Tái hiện Kết quả

Từ thư mục gốc của repository, thực hiện:

```bash
# 1. Chạy toàn bộ unit tests
pytest src/test_agents.py -v

# 2. Chạy hai bộ benchmark so sánh
python src/benchmark.py
```
