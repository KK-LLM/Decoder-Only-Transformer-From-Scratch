# 预训练数据

本项目从 **20 个公开数据来源**中选取语料，经过清洗、去重和采样，构建了约 **20 亿 tokens** 的预训练集，用于 Decoder-only 模型的八轮预训练实验。

## 1. 数据方向与来源

数据按主要内容组织为七个方向，覆盖中英文网页、指令、对话、阅读、数学和代码任务。

| 数据方向 | 使用的数据集 | 主要内容 |
|---|---|---|
| 网页文本 | [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ)、[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 中文为主的互联网文本、英文教育类网页 |
| 通用指令 | [COIG](https://huggingface.co/datasets/BAAI/COIG)、[COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA)、[Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)、[Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)、[OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)、[SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 问答、摘要、改写、翻译及其他指令任务 |
| 通用对话 | [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)、[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) | 合成对话与人工众包的助手对话 |
| 中文对话 | [LCCC](https://huggingface.co/datasets/thu-coai/lccc)、[DuConv](https://github.com/baidu/knowledge-driven-dialogue)、[KdConv](https://github.com/thu-coai/KdConv)、[NaturalConv](https://github.com/naturalconv/NaturalConvDataSet) | 日常闲聊、知识对话和话题驱动的多轮对话 |
| 阅读与常识 | [DuReader](https://github.com/baidu/DuReader)、[CORECODE](https://github.com/danshi777/CORECODE) | 基于资料的问答、对话常识及因果推断 |
| 数学 | [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)、[Ape210K](https://github.com/Chenny0808/ape210k) | 数学应用题、文字推理与程序求解 |
| 代码 | [Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K)、[CodeFeedback-Filtered-Instruction](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 代码生成、程序修改和 SQL 等指令任务 |

本项目根据预训练的数据规模和内容需求，对上述数据集进行筛选与采样，并非全量使用各数据集的所有数据。这样可以控制训练成本，兼顾不同数据方向的覆盖，避免语料过度集中于少数来源。

## 2. 数据选择与筛选

数据处理先从各来源的清洗和格式统一开始：过滤空缺内容、异常长度、明显的 HTML/符号噪声和重复字符，将数据整理为纯文本或对话，并在部分来源内去重。各方向的具体处理如下：

| 数据方向 | 主要处理 |
|---|---|
| 网页文本 | 保留正文 `text`，进行长度和噪声检查，按来源限制进入候选池的规模。 |
| 通用指令 | 将问题、指令和答案统一为 `user` / `assistant` 消息，检查缺失字段、过短回答和格式异常。 |
| 通用对话 | 统一角色；从 OASST1 对话树重建对话并保留中英文内容；移除末尾没有对应回答的用户消息。 |
| 中文对话 | LCCC 处理分词空格并过滤低信息回复；知识对话保留目标、主题或知识信息；无显式角色的部分来源按轮次规则赋予角色。 |
| 阅读与常识 | DuReader 将有限长度的参考资料放入 `system`，过滤缺失答案或无有效证据的样本；CORECODE 将对话、问题及选项组织为输入。 |
| 数学 | Ape210K 对方程与答案进行安全求值比对，过滤明确不一致的样本；无法验证的部分样本保留标记。 |
| 代码 | 将代码问题与回答统一为消息格式，进行长度、噪声检查及去重。 |

随后按来源和数据类型控制采样规模，将对话转换为带角色标记的文本，再进行全局精确去重：对首尾空白、换行及连续空白作轻量归一化后，计算 SHA-256 并去除重复文本。这一步将候选池由 **5,894,883 条**减少至 **5,721,976 条**，去重后的数据进入 token 计数与采样流程。

## 3. 各方向的数据示例

下面从预训练语料中精选 **14 条样例**，每个方向展示 2 条。问题、回答和代码均保留数据集原文，节选内容已标注；为便于阅读，省略训练文本中的角色标记和系统模板。

<details>
<summary>网页文本：2 条样例</summary>

**CCI3-HQ｜藜麦的传统用途（网页节选）**

> 与马铃薯一样,藜麦是前印加时期安第斯人的主要食物之一。传统上,人们将藜麦粒烘焙后制成面粉,并烤制出不同种类的面包。

原网页：[联合国粮农组织：藜麦](https://www.fao.org/quinoa/zh)。

**FineWeb-Edu｜木星与土星的视逆行（网页节选）**

> The planets didn't actually loop by reversing the direction of their orbits, though. Their apparent retrograde motion is a reflection of the motion of the Earth itself. Retrograde motion can be seen each time Earth overtakes and laps planets orbiting farther from the Sun, Earth moving more rapidly through its own relatively close-in orbit.

原网页：[NASA APOD：Jupiter and Saturn Pas de Deux](https://science.nasa.gov/image-article/apod-2001-december-20-jupiter-and-saturn-pas-de-deux/)。

</details>

<details>
<summary>通用指令：2 条样例</summary>

**Infinity-Instruct｜正式措辞改写**

**问题**

> Please convey in professional terminology the following message: This drawing, as indicated, serves as an initial draft and is liable to modifications pending further detailed discussions.

**回答**

> This illustration, as noted, represents a preliminary draft and is subject to revisions based on forthcoming detailed deliberations.

**Infinity-Instruct｜撰写祝福短笺**

**问题**

> Craft a concise and heartfelt message to celebrate a coworker's joyous occasion of welcoming a newborn into their family.

**回答**

> Dear [Coworker's Name],  
> 
> Congratulations on the arrival of your beautiful baby! This is such a joyful and precious time for your family. Wishing you all the love, happiness, and countless cherished moments as you embark on this incredible journey of parenthood. Welcome to the world, little one!  
> 
> Warmest wishes,  
> [Your Name]

</details>

<details>
<summary>通用对话：2 条样例</summary>

**UltraChat-200k｜自然场景描写**

**问题**

> Write a descriptive paragraph about a tranquil lake nestled among towering trees that sway in the gentle breeze, while melodious birdsongs fill the air with their sweet melodies.

**回答**

> As if from a fairytale, a tranquil lake stands nestled amongst towering trees that stretch high into the sky. Each brings with it a flurry of green leaves that sway in the gentle breeze, creating a symphony of rustles and murmurs. The lake glistens as the sunlight pours through the trees and dances on the water's surface in delicate patterns. Melodious birdsongs fill the air with their sweet melodies, adding to the serene atmosphere that radiates tranquility. As you stand there, surrounded by the sounds of nature, you can't help but feel at peace amidst this picturesque scene.

**OASST1｜精简句子**

**问题**

> Can you help make this sentence shorter? "Well.. this might not be like the optimal solution for the way you are looking at it right now you know..."

**回答**

> "This may not be the best solution from your perspective."

</details>

<details>
<summary>中文对话：2 条样例</summary>

**NaturalConv｜相约学打台球（连续四轮节选）**

**用户**

> 说到台球，我以前就在学校里面玩过一点。

**对话方**

> 是啊，我们学校也有开，很多男生都喜欢去玩。

**用户**

> 你玩过吗？

**对话方**

> 没有，我不会啊，我都是在旁边看着的。

**用户**

> 那你想学打台球吗？

**对话方**

> 可以吗？你就可以教我吗？

**用户**

> 当然可以啊，我们下次约个时间，在我们学校见吧。

**对话方**

> 好呀，我也想多学学其他的运动呐。

**NaturalConv｜点单与兴趣交流（连续三轮节选）**

**用户**

> 老板，给我来杯奶茶。

**对话方**

> 好的，加珍珠吗？

**用户**

> 多加一点吧，我喜欢吃珍珠。

**对话方**

> 好的。

**用户**

> 我看你屏幕上放着网球比赛，你平时关注网球吗？

**对话方**

> 我还挺喜欢网球的，没事就看看他们的比赛。

</details>

<details>
<summary>阅读与常识：2 条样例</summary>

**DuReader｜从参考资料提取答案（资料节选）**

**参考资料**

> hm2是面积单位，表示公顷，一般用于土地面积的计算。

**问题**

> hm2是什么单位

**回答**

> 面积单位

原文中的 `hm2` 表示 hm²；单位含义可参见 [NIST 单位表](https://www.nist.gov/pml/special-publication-811/nist-guide-si-chapter-5-units-outside-si)。

**CORECODE｜根据对话判断原因（对话节选）**

**对话资料**

> A: 诶，那你是怎么喜欢上足球的啊？姐姐。  
> B: 哈哈哈，说来话长，因为我老公以前是个足球运动员呢。  
> A: 噢噢，明白啦，因为老公喜欢上的球赛是不是？  
> B: 是啊，以前年轻的时候，我就经常去足球场看他踢球。  
> A: 是不是一看球赛就能想起他踢球的影子啊？  
> B: 嘿嘿，被你猜对啦。就是因为这样我才喜欢看足球的。

**问题**

> 请根据对话内容，直接回答下面问题的答案，不要重述问题或解释原因：事件“B喜欢看足球”的事件原因是什么？

**回答**

> B看球赛能想起老公踢球的影子

</details>

<details>
<summary>数学：2 条样例</summary>

**Ape210K｜数量关系应用题**

**问题**

> 养殖场养鸭376只，养鸡的只数比鸭多258只．这个养殖场一共养鸭和鸡多少只？

**回答**

> 方程：x=376+258+376  
> 答案：1010

**MathInstruct｜两步算术应用题**

**问题**

> Lana had 8 blank pages left in her binder, but she knew she would need more for her next class. Duane took half of the 42 pages in his binder out and gave them to her. How many pages does Lana have in her binder after adding Duane’s?

**回答**

> Duane gave Lana 42 / 2 = 21 pages.  
> After adding Duane’s, Lana has 21 + 8 = 29 pages in her binder.  
> The answer is 29

</details>

<details>
<summary>代码：2 条样例</summary>

**Magicoder-Evol-Instruct-110K｜查找全部匹配索引（Python）**

**题目**

> Write a Python program that takes a list and a search query as input and returns the indices of all elements in the list which match the search query.  
> list = ["apple", "banana", "peach", "orange", "grape"]  
> search_query = "apple"

**回答（代码节选）**

```python
def find_indices(lst, search_query):
 return [i for i, x in enumerate(lst) if x == search_query]

# Test the function
lst = ["apple", "banana", "peach", "orange", "grape"]
search_query = "apple"

print(find_indices(lst, search_query))
```

运行上述原文代码，输出为 `[0]`。

**CodeFeedback-Filtered-Instruction｜条件筛选与平方（Python）**

**题目**

> Use the list comprehension syntax to make a new list whose elements are the squares of the odd numbers greater than 10 in the first list. The first list should contain both even and odd numbers.

**题目给定的输入**

```python
lst = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]
```

**回答**

```python
squared_odd_numbers = [x**2 for x in lst if x > 10 and x % 2 != 0]
print(squared_odd_numbers)
```

先运行题目给定的输入，再运行回答代码，输出为 `[121, 169, 225]`。

</details>

## 4. 最终预训练规模

最终训练集包含 **5,500,992 条源样本，约 20 亿 tokens**；验证集包含 **27,819 条源样本，约 1,000 万 tokens**。

| 用途 | 源样本数 | 打包后的序列数 | 打包后的 token 数 |
|---|---:|---:|---:|
| 训练集（含并入的备用池） | 5,500,992 | 976,560 | 1,999,994,880 |
| 验证集 | 27,819 | 4,882 | 9,998,336 |

表中统计单份语料的规模。源样本是一篇文本或一组对话，序列是打包后的 2,048-token 训练单元；一条序列可以包含多篇文本。

## 5. 训练前处理

编码使用 [SentencePiece BPE tokenizer](../tokenizer/spiece.model)，词表大小为 **48,000**，支持 byte fallback。

1. **真实 token 计数与采样划分**：对每条文本编码，并将末尾的一个 EOS 计入 token 数。以固定 seed `20260710` 分配约 19 亿 tokens 的正式训练集、1,000 万 tokens 的验证集和 1 亿 tokens 的备用池，候选集合中单一来源的 token 占比上限为 25%。确定划分后，编码和打包沿用同一分配结果。
2. **编码和打包**：每条文本末尾追加 EOS，不添加 BOS；在各自划分内连续拼接为 **2,048-token 序列**，丢弃各划分末尾不足长度的余数。

完成打包后，将备用池分片并入训练集，沿用已有的编码与打包结果。最终得到 **120 个训练分片和 1 个验证分片**，验证集不参与参数更新。

## 数据使用说明

- **发布内容**：本仓库提供数据来源、处理流程与精选样例，完整语料和训练分片不随仓库发布。
- **数据划分**：部分来源的上游 train/dev/test 一并进入候选池，训练与验证使用本项目重新划分的结果。
- **处理特性**：精确去重处理完全重复的文本，语义近重复仍可能保留；代码语料未做全量执行验证。文本清洗与 tokenizer 的空白归一化会压缩换行和缩进。
- **数据许可**：已逐一核实训练所用数据集的许可，数据的使用范围与方式均符合相应许可要求。
