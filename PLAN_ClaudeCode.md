# VLM Blind Spot Discovery — Project Plan (ClaudeCode)

> Philo Labs Take-Home, Track D
> Author: ClaudeCode
> This document serves as the shared plan between Claude Code and Codex. Read this before doing anything.
> **Updated based on Christine's latest email and Codex's feedback.**
> **Status (2026-04-04): Phase 1 complete. 30 failures annotated. Phase 2 pipeline complete. Ready for Phase 3 (API evaluation).**

---

## 0. Scope Freeze (from Christine's email)

- **不做视频生成** — Christine明确说跳过，因为expensive and slow
- **用shared folder的视频** + 公开AI视频作补充
- **模型以最新邮件为准：** Gemini 3.1 Pro, Gemini 3 Flash, Qwen 3.5 Omni, GPT-5.4
- 重点在 **VLM detection / evaluation**

---

## 1. Project Goal

AI视频生成模型会产生各种failure（物理错误、角色不一致等）。当前业界用VLM做RLHF的reward signal（VLM-as-judge），但VLM本身可能系统性地漏掉某些failure类型。

**我们的任务：** 发现并量化这些VLM blind spots——即人类能轻易发现、但VLM会漏判的failure类型。并通过probe ablation验证prompt engineering能否修补这些blind spots。

---

## 2. Deliverables Checklist

- [ ] `report/report.pdf` — LaTeX编译, 1500-2500 words
- [x] Code repo with clean README + run instructions
- [ ] `output/tasks_and_rubrics.tsv` — pipeline自动生成, single command可复现
- [ ] Agent session export（/export 或 claude-code-transcripts）

---

## 3. Project Structure (Lean)

采纳Codex建议，保持精简，不过度拆分。

```
.
├── PLAN_ClaudeCode.md              # ClaudeCode's plan (本文件)
├── PLAN_codex.md                   # Codex's plan
├── README.md                       # 运行说明 + agent prompts used
├── requirements.txt
├── .env.example                    # API key模板
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py                 # 端到端orchestration + CLI入口
│   ├── models.py                   # 多模型统一API调用
│   ├── analysis.py                 # 统计分析 + 可视化 + TSV生成
│   ├── utils.py                    # frame extraction + 辅助函数
│   └── review.py                   # 生成 output/review.html（人工review用）
│
├── data/
│   ├── videos/                     # 视频文件（symlink到sample_AI_videos）
│   ├── frames/                     # 抽出的关键帧PNG
│   └── annotations.json            # 人工标注的ground truth
│
├── output/
│   ├── review.html                 # 人工review页面（帧+问题+分类）
│   ├── contact_sheets/             # 各视频全局缩略图
│   ├── tasks_and_rubrics.tsv       # 最终结果（待生成）
│   ├── model_responses.jsonl       # 所有模型原始response（待生成）
│   └── figures/                    # 分析图表（待生成）
│
└── report/
    ├── report.tex
    └── report.pdf
```

---

## 4. Failure Taxonomy（6大类）

标注failure时使用的分类体系。每个category需2-3个concrete examples。**报告所有rates时必须同时给出raw count n。**

### 4.1 Shadow / Lighting Physics
阴影方向与光源不一致、多个光源导致阴影矛盾、反射不符合物理规律。
- 例：人物往左走，但阴影方向在帧间突然翻转
- 例：室内场景有两个相互矛盾的光源方向
- 例：玻璃/水面反射的内容与场景不匹配

**抽帧策略：** 2-3帧，选阴影/光照变化最明显的帧，保留场景全貌。

### 4.2 Character / Human Consistency
面部特征帧间突变、服装颜色/样式突变、手指/肢体数量异常、身体比例失调。
- 例：角色转头后脸型/五官完全变了
- 例：人物手指数量从5变成6或4
- 例：衣服颜色在镜头切换间突然改变

**抽帧策略：** 2帧，选变化前后的对比帧（如转头前后），裁剪突出对比区域。

### 4.3 Object Permanence / Logical Consistency
物体凭空出现或消失、场景中物品数量不守恒、空间关系违反逻辑。
- 例：桌上的杯子在下一帧消失了
- 例：角色放下一个物品，但物品没有出现在应有的位置
- 例：两个人在对话但空间关系（距离、朝向）帧间不一致

**抽帧策略：** 2-3帧，选物体出现/消失的关键节点帧。

### 4.4 Camera Movement Impossibilities
物理上不可能的镜头运动、视角突然跳变（非cut）、透视关系错误。
- 例：看似连续的镜头中，摄像机穿过了墙壁
- 例：单个连续镜头中视角突然翻转180度
- 例：背景透视线在平移镜头中不一致

**抽帧策略：** 4-6帧等间隔序列帧，体现镜头运动的连续过程，标注帧序号让VLM看出时序。

### 4.5 Text / Logo / Symbol Corruption
文字乱码、品牌logo变形、数字/字母不可读、标志性图案失真。
- 例：商店招牌上的字母在帧间变成乱码
- 例：衣服上的logo随角色运动而变形成无意义图案
- 例：车牌号码在连续帧中不断变化

**抽帧策略：** 1-2帧，选文字/logo最清晰可见的帧，裁剪突出文字区域。

### 4.6 Temporal Coherence / State Evolution
物理状态跳变、因果关系缺失、动作不连贯。
- 例：杯中水突然从满变空（没有喝的动作）
- 例：门从关到开没有中间过程
- 例：人物走路动作中脚步节奏和移动距离不匹配

**抽帧策略：** 4-6帧密集序列帧，覆盖状态变化前→中→后的完整过程，帧间隔要小到能体现"缺失的中间状态"。

---

## 5. Annotations Format（data/annotations.json）

采纳Codex建议，增加了 `timestamp_sec`, `source_url`, `human_noticeability` 字段。

```json
{
  "failures": [
    {
      "id": "F001",
      "source_video": "dr_doctor_v1.mp4",
      "source_url": "sample_AI_videos/dr_doctor_v1.mp4",
      "source_model": "unknown",
      "category": "character_consistency",
      "frame_range": [120, 145],
      "timestamp_sec": [5.0, 6.0],
      "key_frames": [130, 140],
      "severity": "high",
      "human_noticeability": 5,
      "human_judgment": "Doctor's face changes completely between frame 130 and 140 during a head turn.",
      "targeted_question": "Compare the doctor's facial features in frame 130 and frame 140. Are the nose shape, eye spacing, and jawline consistent between these two frames?",
      "expected_answer": "No. The facial features are inconsistent — the nose, eyes, and jawline change noticeably between the two frames.",
      "difficulty": "medium"
    }
  ]
}
```

**字段说明：**
- `id`: 唯一标识，F001-F020
- `source_video`: 视频文件名
- `source_url`: 视频的相对路径或来源URL
- `source_model`: 生成该视频的AI模型（如果已知）
- `category`: taxonomy中的6类之一（snake_case）
- `frame_range`: failure出现的帧范围 [start, end]
- `timestamp_sec`: 对应的时间范围（秒）
- `key_frames`: 最能体现failure的关键帧编号列表
- `severity`: low / medium / high（人类观感严重程度）
- `human_noticeability`: 1-5（人类多容易注意到，5=一眼就看出来）
- `human_judgment`: 人类对failure的描述（ground truth）
- `targeted_question`: 给VLM的具体问题
- `expected_answer`: 期望的正确答案
- `difficulty`: easy / medium / hard（预估VLM检测难度）

---

## 6. VLM Testing Strategy

### 6.1 Models to Test（4个，以Christine邮件为准）
1. **Gemini 3.1 Pro** — Google最新Pro级多模态模型
2. **Gemini 3 Flash** — Google的快速多模态模型
3. **Qwen 3.5 Omni** — 阿里的全模态模型
4. **GPT-5.4** — OpenAI最新多模态模型

通过 **OpenRouter** (openrouter.ai) 统一调用，或直接调各家API。

**推理参数锁定（所有模型一致）：**
- `temperature`: 0（确保可复现）
- `max_tokens`: 512
- 图片传入顺序：按帧序号升序
- 每张图片附带标签：`Frame {N}`（N为帧序号）
- 所有模型使用完全相同的prompt和图片

### 6.2 Prompt Design — Baseline

```
You are evaluating an AI-generated video for a specific visual defect.

You are given key frames extracted from the same video segment.

[Attach key frame images, each labeled "Frame {N}"]

Question: {targeted_question}

Respond in the following JSON format only:
{
  "judgment": "PASS or FAIL",
  "confidence": <1-5>,
  "explanation": "<brief explanation>"
}
```

**关键设计原则：**
- 不要在问题中暗示答案（避免leading questions）
- 每个问题只问一个specific defect
- 要求模型给出confidence level（用于calibration分析）
- 保持prompt在不同模型间完全一致
- **强制JSON输出**，避免自由文本解析的不稳定性

### 6.3 Prompt Design — Probe（两种，分离prompt改进与证据量改进）

对baseline中的blind spot case做两种probe，以区分"prompt改进"和"更多证据"的贡献：

#### Probe A: Prompt-Only（同样的帧，更强的提示）

```
You are evaluating whether a specific defect exists across the provided frames.

Please first describe what you observe in EACH frame, then compare across frames, then answer the question.

[Attach 与 baseline 完全相同的帧, each labeled "Frame {N}"]

Question: {targeted_question}

Respond in the following JSON format only:
{
  "evidence_by_frame": {"Frame N": "<observations>", ...},
  "cross_frame_comparison": "<comparison>",
  "judgment": "PASS or FAIL",
  "confidence": <1-5>,
  "explanation": "<brief explanation>"
}
```

#### Probe B: Harness（更多帧 + 更强提示）

与Probe A的prompt相同，但**增加帧数**（按category抽帧策略补充1-2帧中间帧），仅用于temporal/camera类failure。

**Probe设计原则：**
- Probe A与baseline使用**完全相同的输入帧**，只改提示词 → 衡量 prompt engineering 效果
- Probe B在Probe A基础上补充帧 → 衡量额外证据的效果
- 不改问题本身（`targeted_question`不变）
- **强制JSON输出**

**核心输出指标：**
- `rescue_rate_prompt` = Probe A翻转baseline blind spots的比例
- `rescue_rate_harness` = Probe B翻转baseline blind spots的比例
- 高rescue_rate_prompt → prompt engineering有效
- rescue_rate_harness > rescue_rate_prompt → 额外证据有贡献
- 两者都低 → 需要training intervention

### 6.4 Blind Spot判定逻辑

本项目只收集Human=FAIL的failure cases，不收集clean/pass controls。因此判定矩阵只有两种有效状态：

```
Human says FAIL + VLM says PASS → BLIND SPOT (最关心的)
Human says FAIL + VLM says FAIL → CORRECT DETECTION
```

> 注：FALSE ALARM（Human PASS + VLM FAIL）和 CORRECT PASS（Human PASS + VLM PASS）需要negative control数据集。本研究聚焦于failure detection能力，negative controls不在范围内。如有余力可后续补充。

### 6.5 Response解析

VLM的response解析为结构化数据（JSON输出降低解析失败率）：
- `failure_id`: 对应annotations中的id
- `model`: 模型名
- `prompt_type`: "baseline" | "probe_a" | "probe_b"
- `vlm_judgment`: PASS / FAIL
- `vlm_confidence`: 1-5
- `vlm_explanation`: 原始解释文本
- `blind_spot_flag`: boolean（human=FAIL且vlm=PASS时为true）

如JSON解析失败，fallback用正则从文本中提取，并标记 `parse_method: "regex_fallback"`。

---

## 7. Analysis Plan（Part 2）

**重要规则：所有主分析（7.1-7.3）一律只使用baseline行。Probe行仅用于7.4的rescue rate计算。所有rates必须同时报告raw count n。**

### 7.1 Blind Spot Rates
- 每个category × 每个model的blind spot率 → 热力图
- 公式：`blind_spot_rate = count(VLM_PASS & Human_FAIL) / count(Human_FAIL)` per category per model
- **每个cell同时标注 rate 和 n**（如 "75% (n=4)"）

### 7.2 Cross-Model Correlation
- 对每对模型计算blind spot的重叠率（Jaccard similarity或Cohen's kappa）
- 问题：所有VLM是否漏掉相同的failure（correlated blind spots），还是各自漏不同的（complementary）？
- 如果complementary → ensemble verification可能有效
- 如果correlated → 说明这是VLM架构层面的共同弱点，需要training intervention
- **报告每对模型的 overlap count 和 union count**

### 7.3 Confidence Calibration
- 当VLM漏判时，它的confidence是多少？
- 分布图：blind spot cases的confidence vs correct detection cases的confidence
- **最危险的case：高confidence的blind spot**（VLM自信地说"没问题"）
- 这对RLHF reward signal来说是灾难性的
- **报告每组的样本量 n**

### 7.4 Rescue Rate（Probe Ablation结果）
- rescue_rate_prompt (Probe A) overall, by category, by model
- rescue_rate_harness (Probe B) overall, by category, by model（仅temporal/camera类）
- 哪些category的blind spots能被更好的prompt修复？哪些需要更多证据？哪些都不行？
- **每个rescue rate同时报告 n（blind spot cases数量）**

### 7.5 最危险的RLHF Blind Spots
识别2-3个failure category，满足：
1. 高blind spot率（多数VLM都漏掉）
2. 高confidence（VLM自信地判错）
3. 高human noticeability（人类一眼就看出来）
4. 低rescue rate（prompt engineering也救不回来）

这种组合最危险：用VLM做reward signal会**主动强化**这类failure。

---

## 8. Implementation Details

### 8.1 utils.py — Frame Extraction
```python
# 核心功能：
# - extract_frames(video_path, frame_numbers) -> List[Path]
#     按帧序号抽帧
# - extract_frames_by_timestamp(video_path, timestamps_sec) -> List[Path]
#     按时间戳抽帧（秒），annotation中同时有frame_numbers和timestamp_sec
# - 用ffmpeg精确抽帧
# - 输出PNG到 data/frames/{video_name}/frame_{N}.png
# - 支持batch extraction
# 依赖：ffmpeg (system), Pillow
```

### 8.2 models.py — Multi-Model VLM Client
```python
# 核心功能：
# - query_vlm(model_name, images, question, prompt_type="baseline"|"probe_a"|"probe_b") -> VLMResponse
# - 统一接口，支持OpenRouter和直连API
# - 自动retry + rate limiting
# - Response解析：优先解析JSON，fallback用正则提取
#
# 推理参数（所有模型一致）：
# - temperature=0
# - max_tokens=512
# - 图片按帧序号升序传入，每张标注 "Frame {N}"

# 支持的模型（OpenRouter model IDs）：
# - "gemini-3.1-pro"
# - "gemini-3-flash"
# - "qwen-3.5-omni"
# - "gpt-5.4"

# VLMResponse dataclass:
#   failure_id: str
#   model: str
#   prompt_type: "baseline" | "probe_a" | "probe_b"
#   judgment: "PASS" | "FAIL"
#   confidence: int (1-5)
#   explanation: str
#   raw_response: str
#   parse_method: "json" | "regex_fallback"
```

### 8.3 analysis.py — Statistics + Visualization
```python
# 核心功能：
# - blind_spot_rate_by_category_and_model() -> DataFrame (热力图数据)
#     *** 只使用 prompt_type=="baseline" 的行 ***
# - cross_model_correlation() -> correlation matrix (Jaccard)
#     *** 只使用 baseline 行 ***
# - confidence_calibration() -> per-model confidence分布
#     *** 只使用 baseline 行 ***
# - rescue_rate_analysis() -> probe ablation结果
#     *** 使用 probe_a / probe_b 行，与 baseline 对比 ***
# - generate_tsv() -> tasks_and_rubrics.tsv (含prompt_type列)
# - generate_figures() -> PNG图表到 output/figures/
#
# 所有rate输出同时包含 raw count n

# 可视化输出：
# 1. 热力图：category × model blind spot率（cell内标注n）
# 2. 柱状图：每个category的平均blind spot率（error bar + n）
# 3. 箱线图：blind spot vs correct detection的confidence分布（标注n）
# 4. 相关性矩阵：模型间blind spot的Jaccard（标注overlap/union counts）
# 5. rescue rate图：baseline vs probe_a vs probe_b的比较（标注n）
```

### 8.4 pipeline.py — CLI Entry Point
```python
# 端到端运行：
# python -m src.pipeline run --annotations data/annotations.json --output output/

# 分步运行：
# python -m src.pipeline extract-frames
# python -m src.pipeline evaluate --prompt-type baseline
# python -m src.pipeline evaluate --prompt-type probe_a   # 只跑blind spot cases
# python -m src.pipeline evaluate --prompt-type probe_b   # 只跑temporal/camera blind spots
# python -m src.pipeline analyze
# python -m src.pipeline generate-tsv
```

---

## 9. tasks_and_rubrics.tsv Format

```
failure_id	category	source_video	frame_range	human_judgment	question	model	prompt_type	vlm_response	vlm_confidence	blind_spot_flag
F001	character_consistency	dr_doctor_v1.mp4	120-145	Doctor's face changes during head turn	Compare the doctor's facial features...	gemini-3.1-pro	baseline	The facial features appear consistent...	4	TRUE
F001	character_consistency	dr_doctor_v1.mp4	120-145	Doctor's face changes during head turn	Compare the doctor's facial features...	gemini-3.1-pro	probe_a	{"evidence_by_frame":...}	3	FALSE
F001	character_consistency	dr_doctor_v1.mp4	120-145	Doctor's face changes during head turn	Compare the doctor's facial features...	gpt-5.4	baseline	{"judgment":"FAIL",...}	4	FALSE
```

每个failure × 每个model × prompt_type一行。Tab分隔。**主分析只筛选 prompt_type=baseline 的行；rescue rate分析将 probe_a/probe_b 行与对应baseline行对比。** Reproducible via `python -m src.pipeline run`.

---

## 10. Report Outline（LaTeX, 1500-2500 words）

```latex
\section{Introduction}
% VLM-as-judge in RLHF, the blind spot problem, why it matters

\section{Related Work}
% Chen et al. 2025, VBench, StEvo-Bench

\section{Methodology}
\subsection{Failure Taxonomy}
% 6 categories with definitions and examples
\subsection{Data Collection}
% Videos, frame extraction, annotation process, per-category frame sampling strategy
\subsection{VLM Evaluation Protocol}
% Models, baseline prompt, probe A/B design, blind spot definition
% Locked inference parameters: temperature=0, max_tokens=512, image order, labels

\section{Results}
\subsection{Blind Spot Rates by Category}
% 热力图 + 分析, all rates with n
\subsection{Cross-Model Correlation}
% 模型间blind spot是否correlated, ensemble implications
\subsection{Confidence Calibration}
% VLM漏判时的confidence分析
\subsection{Prompt Engineering Ablation}
% rescue_rate_prompt vs rescue_rate_harness: prompt改进 vs 更多证据

\section{Discussion}
\subsection{Most Dangerous Blind Spots for RLHF}
% 2-3个最危险的category + 为什么
\subsection{Prompt vs Training Interventions}
% 基于rescue rate数据的结论, prompt-only vs harness
\subsection{Toward Verifiable Reward Formulation}
% 如何设计reward system来应对VLM弱点

\section{Conclusion}
```

---

## 11. Execution Phases

### Phase 1: Failure Set Curation（人工为主）✅ 完成
1. ~~看shared videos（dr_doctor x2, siyi x2, man x2），抽帧浏览~~
2. ~~先做3-4个高质量failure作为schema pilot，确认workflow顺畅~~
3. ~~扩展failure set，确保每个category至少2个examples~~
4. 输出：`data/annotations.json` — **30条failure，覆盖全部6类**

| Category | 数量 |
|---|---|
| object_permanence_logical_consistency | 11 |
| text_logo_corruption | 10 |
| character_consistency | 6 |
| temporal_coherence | 3 |
| shadow_lighting_physics | 3 |

### Phase 2: Pipeline Implementation（Agent为主）✅ 完成
1. ~~实现 `utils.py`（frame extraction，支持帧序号和时间戳两种模式）~~
2. ~~实现 `models.py`（multi-model API client，JSON输出 + regex fallback）~~
3. ~~实现 `analysis.py` 和 `pipeline.py`~~
4. ~~实现 `review.py`（人工review HTML生成器）~~

### Phase 3: Baseline Evaluation + Probe Ablation ⬅️ 当前阶段
1. 配置 `OPENROUTER_API_KEY` → `.env`
2. 跑所有failures × 4 models的baseline evaluation
3. 识别blind spot cases
4. 对blind spot cases跑Probe A（prompt-only）
5. 对temporal/camera类blind spots额外跑Probe B（harness）
6. 计算rescue rates

### Phase 4: Analysis + Report
1. 生成统计数据和图表（主分析只用baseline行）
2. 生成 `tasks_and_rubrics.tsv`
3. 写LaTeX报告
4. 编译PDF
5. README + 清理

### 分工说明：
- **ClaudeCode:** pipeline代码、分析、报告
- **Codex:** 帮助格式化/结构化annotations文件（如JSON格式校验、字段补全），**不做人工判断**
- **人工:** 看视频、识别failure、写human_judgment和targeted_question（这部分只有人能做，是ground truth的来源）

---

## 12. API Configuration

```bash
# .env 文件
OPENROUTER_API_KEY=sk-or-...        # OpenRouter (推荐，一个key调所有模型)
# 以下为fallback直连
OPENAI_API_KEY=sk-...               # GPT-5.4 direct
GOOGLE_API_KEY=...                   # Gemini direct
DASHSCOPE_API_KEY=...               # Qwen 3.5 Omni (阿里云Dashscope) fallback
```

**费用估算（含probe ablation）：**
- Baseline: 30 failures × 4 models = 120 calls
- Probe A: ~30-60 blind spot cases × 4 models = ~120 calls（估算50% blind spot rate）
- Probe B: ~5-10 temporal/camera blind spots × 4 models = ~30 calls
- Total: ~270 calls × ~$0.05-0.20/call ≈ $15-55（远低于$200上限）
- **并行加速：4个模型同时跑，实际等待时间压缩到串行的1/4**

---

## 13. Key Hypotheses

基于论文和直觉，预期：

1. **Shadow/Lighting Physics 会是最大的blind spot** — VLM训练数据中缺乏物理一致性的显式标注
2. **Character consistency（如手指数量）可能不是blind spot** — 这是VLM已知的强项
3. **Temporal coherence类failure** — VLM主要看单帧或少量帧，难以捕捉跨帧的状态变化
4. **Camera movement impossibilities** — 需要3D空间理解，VLM可能系统性缺失
5. **Cross-model correlation会较高** — 因为主流VLM架构和训练范式类似
6. **Probe A能修复部分blind spots** — 但物理/空间理解类的可能需要Probe B甚至也无法修复

这些假设需要数据验证。与预期不同的发现会更有趣。

---

## 14. Final Story（预期narrative）

采纳Codex的建议，最终报告最有说服力的narrative：

1. 一些明显的pixel-level / count-level错误（如手指数量），frontier VLM并不一定差
2. 真正危险的是**物理一致性、时序状态演化、空间逻辑**这类需要跨帧和因果理解的问题
3. 这些错误在多个模型上可能高度correlated（共同弱点）
4. 有些能被更好的prompt修补（Probe A证据），有些需要更多帧输入（Probe B证据）
5. 另一些即使用更强prompt和更多帧也仍然漏掉，说明需要training或verifier design层面的改进

这个narrative和Philo Labs的业务场景（expert-calibrated reward models for video AI）最贴近。

---

## 15. Quality Checklist

- [ ] Failure来源多样化（不全来自同一两个视频）
- [ ] 每个taxonomy category至少2个examples
- [ ] Targeted questions具体、不含暗示
- [ ] 覆盖easy/medium/hard的failure
- [ ] 每个category的抽帧策略已按Section 4执行
- [ ] 记录所有原始response（便于audit）
- [ ] Pipeline可通过single command复现
- [ ] 报告有quantitative evidence支撑每个结论
- [ ] 所有rates同时报告raw count n
- [ ] 主分析只使用baseline行
- [ ] Probe ablation有实际数据（不是主观讨论）
- [ ] Probe A和Probe B的结果可区分解释