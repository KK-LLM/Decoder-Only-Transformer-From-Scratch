# Decoder-only 预训练复现

使用 PyTorch 手写 **338.61M 参数的 Llama 风格 Decoder-only 模型**，完成中英文数据准备、八轮预训练、KV cache 推理与问答评测。项目公开模型和训练代码、tokenizer、训练日志、最终题库及模型原始回答，展示从模型实现到训练结果的完整过程。

预训练使用约 **20 亿 tokens** 的语料。在本项目的 **200 题中英文诊断集**上，第 7 轮通过 **101/200 题（50.50%）**，为八轮中的最佳结果；第 8 轮回落至 **72/200 题（36.00%）**。下文保留全部轮次的表现，并展示模型答对与答错的真实案例。

- [模型结构](#模型结构)
- [数据准备](#数据准备)
- [预训练过程](#预训练过程)
- [模型评测](#模型评测)
- [后续优化方向](#后续优化方向)

## 模型结构

模型主体由词嵌入、24 个 DecoderBlock、最终 RMSNorm 和语言模型输出层组成。每个 DecoderBlock 使用 **Pre-RMSNorm、RoPE、GQA、SwiGLU 和残差连接**，输入词嵌入与输出层共享权重。

| 配置 | 数值 |
| --- | ---: |
| 参数量 | 338,609,152 |
| 层数 | 24 |
| 隐藏维度 | 1,024 |
| Query heads / KV heads | 16 / 4 |
| FFN 中间维度 | 3,072 |
| 词表大小 | 48,000 |
| 上下文长度 | 2,048 tokens |

[model.py](decoder-only-pretrain/scripts/model.py) 包含上述模块、因果注意力及 next-token 损失计算，注意力算子调用 PyTorch SDPA。[generation.py](decoder-only-pretrain/scripts/generation.py) 实现 KV cache 和 beam search，支持完整 prompt 的 prefill 与随后逐 token 解码。

## 数据准备

预训练语料选自 **20 个公开数据来源**，包括 CCI3-HQ、FineWeb-Edu、Firefly、OpenHermes-2.5、UltraChat-200k、MathInstruct、Ape210K 和代码指令数据，覆盖中英文网页、问答、对话、阅读、数学与代码。

处理流程为：**来源清洗与格式统一 → 精确去重 → token 计数与采样划分 → 编码与打包**。使用词表为 48,000 的 SentencePiece BPE tokenizer，每条文本末尾追加 EOS，再连续打包为 2,048-token 序列。

| 划分 | 源样本数 | 打包序列数 | token 数 |
| --- | ---: | ---: | ---: |
| 训练集 | 5,500,992 | 976,560 | 1,999,994,880 |
| 验证集 | 27,819 | 4,882 | 9,998,336 |

表中是单份语料的规模，八轮训练重复使用同一份训练语料。数据集来源、筛选方法、真实样例和完整处理流程见 [data/README.md](data/README.md)；完整训练语料和分片不随仓库发布。

## 预训练过程

模型从随机初始化开始进行 next-token 预测训练。实际实验使用 4 卡 DDP，共完成八轮训练，配置如下：

| 设置 | 配置 |
| --- | --- |
| 训练方式与精度 | 4 卡 DDP，BF16 混合精度 |
| 每卡 batch / 全局 batch | 8 / 32 |
| 优化器 | AdamW，β₁=0.9，β₂=0.95 |
| Weight decay / 梯度裁剪 | 0.1 / 最大梯度范数 1.0 |
| 随机种子 | 20260712 |

训练分三个阶段，后两个阶段从前一阶段的 checkpoint 继续：

| 轮次 | 学习率安排 | 数据采样 |
| --- | --- | --- |
| 第 1～4 轮 | warmup 2,000 步至 `3e-4`，随后余弦衰减至 `3e-5` | 单分片顺序训练 |
| 第 5～6 轮 | 重新 warmup 1,000 步至 `1.2e-4`，随后余弦衰减至 `1e-5` | 每 8 个分片组成混合池 |
| 第 7～8 轮 | 固定学习率 `1e-5` | 沿用 8 分片混合 |

[training.log](results/pretrain/logs/training.log) 保存训练步数、loss、学习率、累计 token 数、验证结果和训练中的生成示例。训练代码提供[单卡入口](decoder-only-pretrain/scripts/train.py)和 [DDP 入口](decoder-only-pretrain/scripts/train_ddp.py)，包含梯度累积、学习率调度、checkpoint 保存及恢复校验。

### 训练入口

推荐使用 Python 3.12，在仓库根目录安装依赖：

```bash
python -m pip install "torch==2.8.0" "sentencepiece==0.2.2" "numpy>=1.26,<3"
```

运行训练需要支持 BF16 的 CUDA GPU，以及自行准备的训练数据。数据须使用仓库中的 tokenizer 编码，提供 `tokenized_manifest.json` 和 `train/`、`valid/` 分片；分片的 `input_ids` 为 `int32`、形状为 `[N, 2048]`，并带有对应的 `split` 字段。

安装依赖后，在仓库根目录启动初始四轮训练：

```bash
torchrun --standalone --nproc_per_node=4 decoder-only-pretrain/scripts/train_ddp.py \
  --manifest /path/to/pretrain_data/tokenized_manifest.json \
  --output-dir outputs/pretrain_initial \
  --batch-size 8 --global-batch-size 32 --mix-shards 1 --epochs 4 --no-resume
```

将 `--manifest` 替换为实际数据路径。后续阶段使用 DDP 脚本的续训与学习率阶段参数；本条命令对应第 1～4 轮。

## 模型评测

本项目使用 **200 题中英文诊断集**评测第 1～8 轮预训练 checkpoint，比较训练过程中的答题表现，选择表现最好的 checkpoint，并确定后续优化重点。

**第 7 轮表现最好，通过 101/200 题，通过率为 50.50%。** 相比第 1 轮的 21/200，模型能够完成的任务明显增加；第 8 轮回落至 72/200（36.00%）。以下统一用通过题数和通过率展示全部轮次的结果，再通过具体问答说明模型的能力与不足。

### 评测任务与条件

#### 题库构建

本项目围绕七类基础任务构建了 200 道中英文诊断题。其中，**168 道题依据项目验证语料中的样本改写**，材料来源包括 OpenHermes-2.5、FineWeb-Edu、Firefly、MathInstruct、CodeFeedback 等数据集；另外编写 **32 道日常场景题**，包括 16 道阅读题和 16 道对话题，用于观察模型对简单信息提取和日常回应的掌握情况。每道题配有参考答案和评分要点。

最终题库中的 **12 道翻译题包含在上述 168 道题内**，由原有 24 道翻译题根据八个 checkpoint 的已有表现筛选保留，英译中与中译英各 6 道。全部 checkpoint 使用同一版题库进行比较。

题目语言分布为英文题 99 道、中文题 89 道，另有英译中和中译英各 6 道。

| 类别 | 题数 | 主要任务 |
| --- | --- | --- |
| 通识知识问答 | 42 | 基础事实与概念 |
| 阅读理解 | 52 | 依据给定材料提取信息 |
| 自然交流与对话理解 | 52 | 情境回应、人物关系与对话信息 |
| 基础数学 | 18 | 算术和数量关系 |
| 代码理解 | 18 | 代码输出、功能及条件判断 |
| 摘要与改写 | 6 | 概括、简化和保持原意的改写 |
| 基础翻译 | 12 | 英译中与中译英 |

全部 200 题采用单轮问答，每题生成一次回答。“对话理解”题将多人对话作为输入材料，模型据此回答一个问题。题目、参考答案和评分要点见 [questions.jsonl](decoder-only-pretrain/evaluation/training_aligned_bank_v3_review/questions.jsonl)。

#### 生成条件

| 设置 | 统一评测配置 |
| --- | --- |
| 模型 | 同一预训练主线第 1～8 轮 checkpoint |
| 运行设备与精度 | CPU、FP32、4 线程 |
| 随机种子 | 0 |
| 解码 | beam size = 1，使用 KV cache，length penalty = 0 |
| 最大新增 token 数 | 每题 256 |
| 重复惩罚 | 1.0，即未启用额外重复惩罚 |
| tokenizer | 项目使用的 SentencePiece 模型 |
| 输入模板 | 带 system 的统一模板，system 内容为“你是一个有帮助的中英文助手。” |
| 停止条件 | 生成 EOS / END，或达到新增 token 上限 |

模型输入由 system 提示和题目组成，参考答案与评分要点用于生成后的判分。

每轮最终结果合并了 **116 条重测回答与 84 条复用回答**：阅读 52 题、对话 52 题、翻译 12 题重新生成；通识、数学、代码、摘要改写共 84 题复用上一版评测结果。八轮合计 **1,600 条回答记录**。

### 如何判定通过

评分采用 **大模型逐题语义评阅**，对照题目、参考答案和评分要点检查完整回答。

**回答正确，并完成题目要求，判为通过。** 回答应满足全部关键要点，接受同义表达和合理的措辞变化。关键错误、实质性错误解释、影响回答的编造或矛盾，以及角色串写、持续循环、任务未完成等情况，判为未通过。

判分结合具体任务：阅读题明确回答目标信息即可，复述一两句短材料也可以通过；摘要题需要完成概括。达到输出上限时，根据任务是否完成判分。

**通过率 = 通过题数 ÷ 评测题数。** 每轮总表以 200 题为分母，分类结果以该类题数为分母。

新增 32 题对应的 256 条回答按题混排，隐藏 checkpoint 身份后判分。其余题目沿用原评分：其中 84 道重测旧题的回答文本、生成 token 和停止原因与前次一致，另外 84 道题直接复用前次回答。

### 八轮评测结果

所有轮次使用同一最终版 200 题口径，通过率保留两位小数。

| 训练轮次 / Checkpoint | 通过题数 | 通过率 |
| --- | --- | --- |
| 1 | 21/200 | 10.50% |
| 2 | 26/200 | 13.00% |
| 3 | 57/200 | 28.50% |
| 4 | 51/200 | 25.50% |
| 5 | 80/200 | 40.00% |
| 6 | 79/200 | 39.50% |
| 7 | 101/200 | 50.50% |
| 8 | 72/200 | 36.00% |

从第 1 轮到第 7 轮，通过率由 10.50% 提升至 50.50%。提升过程中存在波动：第 4 轮低于第 3 轮，第 6 轮略低于第 5 轮；第 8 轮也没有继续提高，反而回落至 36.00%。因此，本项目选择第 7 轮作为本次评测中表现最好的 checkpoint。

#### 七类任务的通过题数

`C1`～`C8` 表示第 1～8 轮 checkpoint；每格为“通过题数 / 该类题数”。

| 类别（题数） | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 通识知识问答（42） | 0/42 | 1/42 | 7/42 | 4/42 | 16/42 | 17/42 | 17/42 | 8/42 |
| 阅读理解（52） | 8/52 | 16/52 | 27/52 | 21/52 | 31/52 | 26/52 | 38/52 | 38/52 |
| 自然交流与对话理解（52） | 11/52 | 6/52 | 20/52 | 19/52 | 26/52 | 22/52 | 29/52 | 19/52 |
| 基础数学（18） | 0/18 | 2/18 | 1/18 | 1/18 | 1/18 | 4/18 | 6/18 | 1/18 |
| 代码理解（18） | 2/18 | 1/18 | 0/18 | 1/18 | 4/18 | 4/18 | 4/18 | 3/18 |
| 摘要与改写（6） | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 1/6 | 0/6 |
| 基础翻译（12） | 0/12 | 0/12 | 2/12 | 5/12 | 2/12 | 6/12 | 6/12 | 3/12 |

第 7 轮在阅读、对话理解和通识问答中分别通过 38/52、29/52 和 17/42。第 8 轮阅读仍为 38/52，但对话下降至 19/52，数学从 6/18 降至 1/18。不同任务的变化趋势并不一致，数学、代码、摘要改写和输出要求遵循仍是后续优化的重点。

### 模型回答示例

以下示例均来自本次评测中表现最好的**第 7 轮 checkpoint**，展示模型在不同任务中的实际回答。

#### 基础知识问答

题目 ID：`DIST-GENERAL-004`

**问题**

```text
What does the aperture of a camera lens control about the light entering the camera?
```

**模型回答**

```text
The aperture of a camera lens controls the amount of light entering the camera. 
```

#### 阅读理解

题目 ID：`DIST-READING-006`

**问题**

```text
Read the passage and answer from it.
Passage: Fading Earth Species Today, or F.E.S.T, is a website about endangered animals and their habitats. It explains which animals are endangered, how they become endangered, and what people can do to help them.
Question: What is the main subject of this website?
```

**模型回答**

```text
The main subject of this website is the endangered animals and their habitats. 
```

#### 对话理解

题目 ID：`DIST-DIALOGUE-024`

**问题**

```text
Read this conversation.
A: Here is a one-bedroom apartment for 400 dollars. B: Let me find a two-bedroom apartment. Here is one on South Street for 415 dollars. A: Did you say two bedrooms? B: Yes. A: Let us go and have a look.
Question: How many bedrooms does the apartment they want to view have?
```

**模型回答**

```text
The apartment they want to view has two bedrooms. 
```

#### 日常交流

题目 ID：`DIST-DIALOGUE-017`

**问题**

```text
朋友说“最近一直在认真学习”，请接着这个话题问一个自然的问题。
```

**模型回答**

```text
你最近在学习什么? 
```

#### 代码理解

题目 ID：`DIST-CODE_UNDERSTANDING-013`

**问题**

```text
Choose the SQL condition for students who are older than 25 AND have GPA above 3.5. A: age > 25 AND gpa > 3.5. B: age > 25 OR gpa > 3.5.
```

**模型回答**

```text
The SQL condition for students who are older than 25 and have GPA above 3.5. is: A: age > 25 AND gpa > 3.5 
```

#### 基础计算

题目 ID：`DIST-MATH-001`

**问题**

```text
Calculate 0.4 multiplied by 0.6.
```

**模型回答**

```text
0.40 * 0.6 = 0.24 
```

#### 英译中

题目 ID：`DIST-TRANSLATION-006`

**问题**

```text
Translate into Chinese: Computers are now an essential part of modern life.
```

**模型回答**

```text
计算机已经成为现代生活不可或缺的一部分。 
```

**训练效果：** 这些回答表明，模型经过预训练，已经初步具备基础知识问答、短文信息提取、对话理解与回应、简单翻译和条件判断等能力，能够针对中英文输入生成与问题相符的回答。例如，模型能识别租房对话中的最终选择、区分 SQL 的 AND 与 OR 条件，并保留译文的关键含义，体现了本次预训练在语言理解与表达上的效果。

下面的回答展示了同一 checkpoint 在计算、信息定位和摘要任务中仍存在的问题。

#### 数学应用题

题目 ID：`DIST-MATH-013`

**问题**

```text
把20.4米长的绳子平均分成4段，每段长多少米？
```

**模型回答**

```text
方程:x=20.4/4 答案:6.1 
```

#### 阅读理解

题目 ID：`DIST-READING-EASY-042`

**问题**

```text
Read the passage and answer from it.
Passage: Ben goes to school by bus. He sits near a window.
Question: How does Ben go to school?
```

**模型回答**

```text
Ben sits near a window. 
```

#### 文本摘要

题目 ID：`DIST-SUMMARY_REWRITE-002`

**问题**

```text
Summarize this passage briefly: The US and New York state governments are working on the Hudson River Greenway, a bike path and walking trail along the river in New York City. They aim to improve access for pedestrians and cyclists, make the trail safer, and connect it to parks and green spaces.
```

**模型回答**

```text
The US and New York state governments are working on the Hudson River Greenway, a bike path and walking trail along the river in New York City. They aim to improve access for pedestrians and cyclists, make the trail safer, and connect it to parks and green spaces. 
```

**问题分析：** 模型在计算准确性、信息定位和指令执行上仍不稳定：绳子题列出了正确的除法式，却将结果算成了 6.1，正确值应为 5.1 米；阅读题回答了“坐在哪里”，没有回答“怎样上学”；摘要题则直接复述原文，没有完成概括。这些表现说明，模型还需要更可靠地执行运算、定位所问信息并遵循任务要求，后续可通过针对性的数据补充和 SFT 验证改进效果。

八轮共 1,600 条完整回答见 [merged_inputs.json](results/pretrain/merged_inputs.json)，可按 `epoch` 和 `question_id` 查找。

### 运行评测

将 `checkpoint_epoch_0007.pt` 放入 `weights/pretrain/`，在项目根目录运行：

```bash
python decoder-only-pretrain/scripts/evaluate_checkpoint.py \
  --checkpoint weights/pretrain/checkpoint_epoch_0007.pt \
  --output-dir outputs/pretrain_evaluation_epoch7
```

生成的回答保存在 `outputs/pretrain_evaluation_epoch7/`，评分按前文方法另行进行。

## 后续优化方向

后续将从**数据、SFT 与模型架构**三个方向推进，重点提高回答准确性、问题理解和信息利用能力。

- **数据质量与任务覆盖：** 补充小数运算、单位换算和多步数量关系数据，强化计算与条件理解；增加包含干扰信息、实体与关系变化的阅读材料，改善目标信息定位，并逐步扩展到更长文本。

- **SFT 与指令遵循：** 围绕摘要、改写、多约束指令和自然交流开展专项训练，改善复述原文、遗漏要求、重复生成及结束不及时等问题。同时检查基础问答、翻译等已有能力是否保持，兼顾任务表现与能力保留。

- **模型架构与信息利用：** 已系统调研归一化、Value Residual、短卷积及 GDN/SSM 与 Attention 混合等方向，并实现一版新架构候选：由 **338.61M、24 层的纯 Attention＋MLP**，调整为 **384.23M、32 层、hidden=768 的 Attention＋Mamba-1 SSM 并联结构**，保留 MLP，并加入 **Peri-RMSNorm 和可学习 Value Residual**。

## 项目结构

主要文件如下：

```text
Decoder-only-github/
├── .gitignore                            # 本地文件及生成产物排除规则
├── README.md
├── data/README.md                         # 数据来源与处理流程
├── tokenizer/spiece.model                 # SentencePiece tokenizer
├── decoder-only-pretrain/
│   ├── scripts/
│   │   ├── model.py                       # 模型实现
│   │   ├── generation.py                  # KV cache 与生成
│   │   ├── train.py                       # 单卡训练
│   │   ├── train_ddp.py                    # DDP 训练
│   │   └── evaluate_checkpoint.py         # 评测回答生成
│   └── evaluation/
│       └── training_aligned_bank_v3_review/questions.jsonl
└── results/pretrain/
    ├── logs/training.log                  # 训练日志
    └── merged_inputs.json                # 八轮模型原始回答
```

## 参考资料

- 模型结构：[Llama 2](https://arxiv.org/abs/2307.09288)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[RoPE](https://arxiv.org/abs/2104.09864)、[GQA](https://arxiv.org/abs/2305.13245)、[SwiGLU](https://arxiv.org/abs/2002.05202)。
- 分词工具：[SentencePiece](https://github.com/google/sentencepiece)。
- 注意力算子：[PyTorch SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)。
- 数据集来源与链接：[数据说明](data/README.md)。
