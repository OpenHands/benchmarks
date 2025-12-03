旧版本观察到的问题

  1. Trajectory太长：每个trajectory有80+步，包含大量的thinking、action、observation
  2. 预处理过于简单：当前TrajectoryFormatter只是简单截断到30步，没有智能筛选
  3. 提取的workflow太泛化：当前的4个workflow（如"Locate Relevant Code", "Fix and Verify"）是任何bug-fixing任务都适用的通用模板，缺乏对specific coding patterns的洞察
  4. 没有跨trajectory的聚类：相似问题的trajectory没有被归类在一起进行联合分析
                                                                                                                                                                                                               
⏺ User answered Claude's questions:
  ⎿  · 对于Trajectory预处理，您倾向于哪种压缩策略？ → 可以模块化的enable （也就是这个compression 
     module，或者说class，需要同时留有关键步骤提取，层次化摘要，action过滤的接口。由于我们可以发现当前我们的这三个方法，第一个关键部提取，第二个层次化摘要，以及第三个action类型过滤，这三个方法它们是相互冲突的
     ，所以说这三个method或者说这三个模式它们都是平行的，它们不能够相互兼容。但是我们仍然要模块化的设计这一整个compression module，这样可以为了方便我们在未来进行一些功能的叠加，我们之后可能会比如说层次化摘要
     结合另外一个方法，就是method叠加method再叠加method，那这么做的前提就是我们必须得有一个完全模块化的class，一定要是一个class，这个compression
     module必须是一个class，它不能只是一个函数，它必须是一个完整的Python文件，一整个class。
     · 您希望提取的workflow达到什么粒度的specificity？ → 两级结构，且必须是class - scope 的设计，方便enale 和 disbale 
     workflow的提取方法。并且更关键的一点就是我们这个workflow的或者说叫做induction的这个class这个induction的module它需要支持大语言模型的它是需要严重的依赖大语言模型的一个提取的所以我不管就是你最终这个结构到底
     是什么结构是两级结构还是一个什么什么什么结构但总之的话呢你在写这个induction的class的时候你是要有这个接口的要有一个可维护可扩展的这个大语言模型的接口让大语言模型这个open
     router的大语言模型来进行一个提取那至于这个输出结构对吧那这个东西也是需要就是它是一个restriction是放在这个大语言模型所提取的内容的基础之上的对吧因为我们给prompt也会给到大语言模型说它必须得按照这样一个结构
     来输出对吧大概是这么一个意思就是说你这个结构的这些信息你需要传给大语言模型然后大语言模型就知道我到底是用两级结构还是什么代码模式还是说问题类型的这种这种级别那总之呢你这个induction的这个模块需要是一个需要
     是一个完整的class然后它可以很灵活的来选取就是我们要用哪种结构那选取哪种结构也就取决于大语言模型最后要怎么去进行一个induction
     · 对于相似trajectory的聚类，您希望基于什么维度？ → 需要同时支持问题描述相似性或action序列相似性或代码修改相似性。就同理,聚类维度这个东西,这个clustering的这个module它也必须要是一个class。那这个class的话呢
     ,它就可以很灵活地去支持,对吧?以上我说的这三个相似性的计算方法,对吧?它有这样的method,这种相似性的一个method,那你传参数的时候你就知道method的similarity
     calculator,对吧?它的method是等于问题描述相似性呢,还是action呢,还是说代码修改相似性呢,对吧?这样子才是一个可维护可扩展的这么一个写法。
     · 您希望新的induction模块输出什么样的workflow格式？ → 保持现有格式

---

# CAWM Induction Module 升级计划

## 目标
为OpenHands benchmarks项目设计模块化的CAWM（Code Agent Workflow Memory）系统，包含三个可组合的class：CompressionModule、ClusteringModule、InductionModule。

## 新增文件结构

```
CAWM/
├── models.py          # 共享数据模型（新建）
├── llm_client.py      # OpenRouter兼容的LLM客户端（新建）
├── compression.py     # CompressionModule（新建）
├── clustering.py      # ClusteringModule（新建）
├── induction.py       # InductionModule（新建）
├── pipeline.py        # Pipeline编排器（新建）
├── __init__.py        # Package导出（更新）
└── [现有文件保持不变]
```

---

## 实现步骤

### Step 1: 创建 `CAWM/models.py`
核心数据模型：
- `ActionType` (Enum): exploration, file_view, file_edit, testing, terminal, etc.
- `WorkflowStep` (dataclass): 保持现有格式 (env_description, reasoning, action, action_type)
- `Workflow` (dataclass): 增加 `level` 字段支持两级结构
- `TrajectoryEvent` (dataclass): 解析后的事件结构
- `Trajectory` (dataclass): 完整trajectory包装
- `TrajectoryCluster` (dataclass): 聚类结果

### Step 2: 创建 `CAWM/llm_client.py`
OpenRouter兼容的LLM客户端：
```python
class LLMClient:
    def __init__(self, provider="openrouter", model="anthropic/claude-3.5-sonnet", ...)
    def complete(self, prompt: str, system: str = None) -> str
    def parse_structured_response(self, response: str) -> Any
```
- 支持 openrouter, openai, anthropic 三种provider
- 环境变量fallback: OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

### Step 3: 创建 `CAWM/compression.py`
CompressionModule - 三种互斥策略（Strategy Pattern）：

```python
class CompressionStrategy(Enum):
    KEY_STEP_EXTRACTION = "key_step_extraction"
    HIERARCHICAL_SUMMARIZATION = "hierarchical_summarization"
    ACTION_TYPE_FILTERING = "action_type_filtering"

class CompressionModule:
    def __init__(self, strategy: str, llm_client: LLMClient = None, config: CompressionConfig = None)
    def compress(self, trajectory: Trajectory) -> Trajectory
    def compress_batch(self, trajectories: List[Trajectory]) -> List[Trajectory]
    def __add__(self, other) -> ComposedCompression  # 支持组合
```

**策略实现：**
1. `KeyStepExtractionStrategy`: 提取导致代码修改的关键步骤
2. `HierarchicalSummarizationStrategy`: 用LLM分段摘要（需要llm_client）
3. `ActionTypeFilteringStrategy`: 按action类型过滤

### Step 4: 创建 `CAWM/clustering.py`
ClusteringModule - 三种相似性计算方法：

```python
class SimilarityMethod(Enum):
    PROBLEM_DESCRIPTION = "problem_description"
    ACTION_SEQUENCE = "action_sequence"
    CODE_MODIFICATION = "code_modification"

class ClusteringModule:
    def __init__(self, method: str, llm_client: LLMClient = None, config: ClusteringConfig = None)
    def cluster(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]
    def get_similarity(self, traj1: Trajectory, traj2: Trajectory) -> float
```

**相似性实现：**
1. `ProblemDescriptionSimilarity`: 基于issue description的语义embedding相似性
2. `ActionSequenceSimilarity`: 基于action type序列的N-gram Jaccard相似性
3. `CodeModificationSimilarity`: 基于git_patch的代码修改模式相似性

### Step 5: 创建 `CAWM/induction.py`
InductionModule - 两级workflow结构，重度依赖LLM：

```python
class WorkflowLevel(Enum):
    GENERAL = 1   # 跨项目通用
    SPECIFIC = 2  # 项目/问题类型特定

class InductionModule:
    def __init__(self, llm_client: LLMClient, config: InductionConfig = None)
    def induce(self, trajectories: List[Trajectory], level: WorkflowLevel) -> List[Workflow]
    def induce_from_clusters(self, clusters: List[TrajectoryCluster], level: WorkflowLevel) -> List[Workflow]
    def induce_hierarchical(self, trajectories: List[Trajectory]) -> Dict[WorkflowLevel, List[Workflow]]
```

**Prompt设计：**
- Level 1 (General): 高抽象，跨项目适用，使用placeholder
- Level 2 (Specific): 保留项目/问题上下文，更具体的代码模式

### Step 6: 创建 `CAWM/pipeline.py`
Pipeline编排器：

```python
class CAWMPipeline:
    def __init__(self, llm_client: LLMClient, config: PipelineConfig = None)
    def run(self, trajectories: List[Trajectory]) -> List[Workflow]
    def run_from_file(self, input_path: str, output_path: str = None) -> List[Workflow]
```

Pipeline流程：`Compress → Cluster → Induce`

### Step 7: 更新 `CAWM/__init__.py`
导出所有公共接口

---

## 关键设计决策

1. **模块化**: 每个模块可独立使用或通过Pipeline组合
2. **Strategy Pattern**: Compression和Clustering使用策略模式，方便扩展
3. **LLM依赖**: InductionModule强依赖LLM，Compression的HierarchicalSummarization也需要LLM
4. **向后兼容**: 保持现有WorkflowStep格式，现有代码可继续工作
5. **可组合压缩**: CompressionModule支持`+`操作符进行策略叠加

---

## 示例用法

```python
from CAWM import CAWMPipeline, LLMClient, PipelineConfig, CompressionStrategy

# 初始化
llm = LLMClient(provider="openrouter", model="anthropic/claude-3.5-sonnet")

# 完整Pipeline
pipeline = CAWMPipeline(llm_client=llm)
workflows = pipeline.run_from_file(
    "CAWM/trajectories/resolved_trajectories.jsonl",
    output_path="CAWM/workflow/new_workflows.json"
)

# 自定义配置
config = PipelineConfig(
    compression_strategy=CompressionStrategy.KEY_STEP_EXTRACTION,
    clustering_method=SimilarityMethod.ACTION_SEQUENCE,
    workflow_level=WorkflowLevel.SPECIFIC
)
workflows = pipeline.run_from_file(input_path, config=config)
```

---

## 关键文件路径

| 文件 | 作用 |
|------|------|
| `/Users/tangyiq/dev/benchmarks/CAWM/models.py` | 核心数据模型（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/llm_client.py` | LLM客户端（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/compression.py` | 压缩模块（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/clustering.py` | 聚类模块（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/induction.py` | 归纳模块（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/pipeline.py` | Pipeline编排（新建）|
| `/Users/tangyiq/dev/benchmarks/CAWM/__init__.py` | Package导出（更新）|