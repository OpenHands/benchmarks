# Stage 1-3: 压缩模块 (CAWM/compression.py)

## 目标
创建CompressionModule，支持三种互斥的压缩策略，可模块化组合。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/compression.py`

---

## 枚举和配置

### CompressionStrategy (Enum)
```python
class CompressionStrategy(Enum):
    """压缩策略枚举"""
    KEY_STEP_EXTRACTION = "key_step_extraction"
    HIERARCHICAL_SUMMARIZATION = "hierarchical_summarization"
    ACTION_TYPE_FILTERING = "action_type_filtering"
```

### CompressionConfig (dataclass)
```python
@dataclass
class CompressionConfig:
    """压缩配置"""
    # 通用配置
    max_events: int = 50                 # 压缩后最大事件数
    preserve_first_n: int = 3            # 保留前N个事件
    preserve_last_n: int = 3             # 保留后N个事件

    # KeyStepExtraction配置
    key_action_types: List[ActionType] = field(default_factory=lambda: [
        ActionType.FILE_EDIT,
        ActionType.TESTING
    ])
    include_preceding_context: int = 1   # 关键步骤前保留N个事件
    include_following_context: int = 0   # 关键步骤后保留N个事件

    # HierarchicalSummarization配置
    segment_size: int = 10               # 每段事件数
    summary_max_tokens: int = 200        # 每段摘要最大token
    summarization_prompt: Optional[str] = None  # 自定义摘要prompt

    # ActionTypeFiltering配置
    keep_action_types: List[ActionType] = field(default_factory=lambda: [
        ActionType.FILE_VIEW,
        ActionType.FILE_EDIT,
        ActionType.TESTING,
        ActionType.EXPLORATION
    ])
    exclude_action_types: List[ActionType] = field(default_factory=list)
    deduplicate_consecutive: bool = True  # 去除连续重复的相同类型
```

---

## 策略基类

```python
from abc import ABC, abstractmethod

class BaseCompressionStrategy(ABC):
    """压缩策略基类"""

    def __init__(self, config: CompressionConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client

    @abstractmethod
    def compress(self, trajectory: Trajectory) -> Trajectory:
        """压缩单个trajectory"""
        pass

    def compress_batch(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """批量压缩"""
        return [self.compress(t) for t in trajectories]

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    def requires_llm(self) -> bool:
        """是否需要LLM"""
        return False
```

---

## 策略实现

### 1. KeyStepExtractionStrategy
```python
class KeyStepExtractionStrategy(BaseCompressionStrategy):
    """
    关键步骤提取策略

    提取导致代码修改的关键步骤：
    - FILE_EDIT 事件
    - TESTING 事件
    - 关键步骤前的上下文（exploration, file_view）
    """

    @property
    def name(self) -> str:
        return "key_step_extraction"

    def compress(self, trajectory: Trajectory) -> Trajectory:
        events = trajectory.events
        if len(events) <= self.config.max_events:
            return trajectory

        key_indices = self._find_key_indices(events)
        selected_indices = self._expand_with_context(key_indices, len(events))

        # 保留首尾
        selected_indices.update(range(self.config.preserve_first_n))
        selected_indices.update(range(len(events) - self.config.preserve_last_n, len(events)))

        # 限制数量
        selected = sorted(selected_indices)
        if len(selected) > self.config.max_events:
            selected = self._prioritize_selection(events, selected)

        compressed_events = [events[i] for i in selected]

        return Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=compressed_events,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            metadata={**trajectory.metadata, "compression": self.name, "original_length": len(events)}
        )

    def _find_key_indices(self, events: List[TrajectoryEvent]) -> Set[int]:
        """找出关键事件的索引"""
        key_indices = set()
        for i, event in enumerate(events):
            if event.action_type in self.config.key_action_types:
                key_indices.add(i)
            elif self._is_key_by_content(event):
                key_indices.add(i)
        return key_indices

    def _is_key_by_content(self, event: TrajectoryEvent) -> bool:
        """根据内容判断是否为关键事件"""
        # 检查是否包含重要发现
        if event.observation_content:
            # 错误发现、测试结果等
            important_patterns = [r'error', r'failed', r'passed', r'found', r'fixed']
            for pattern in important_patterns:
                if re.search(pattern, event.observation_content, re.IGNORECASE):
                    return True
        return False

    def _expand_with_context(self, key_indices: Set[int], total_length: int) -> Set[int]:
        """扩展关键索引以包含上下文"""
        expanded = set()
        for idx in key_indices:
            # 添加前文
            for i in range(max(0, idx - self.config.include_preceding_context), idx + 1):
                expanded.add(i)
            # 添加后文
            for i in range(idx, min(total_length, idx + self.config.include_following_context + 1)):
                expanded.add(i)
        return expanded

    def _prioritize_selection(self, events: List[TrajectoryEvent], indices: List[int]) -> List[int]:
        """优先级选择以限制数量"""
        # 优先级: FILE_EDIT > TESTING > FILE_VIEW > EXPLORATION > others
        priority = {
            ActionType.FILE_EDIT: 0,
            ActionType.TESTING: 1,
            ActionType.FILE_VIEW: 2,
            ActionType.EXPLORATION: 3
        }

        scored = [(i, priority.get(events[i].action_type, 10)) for i in indices]
        scored.sort(key=lambda x: x[1])

        return sorted([x[0] for x in scored[:self.config.max_events]])
```

### 2. HierarchicalSummarizationStrategy
```python
class HierarchicalSummarizationStrategy(BaseCompressionStrategy):
    """
    层次化摘要策略

    使用LLM对trajectory分段进行摘要
    """

    @property
    def name(self) -> str:
        return "hierarchical_summarization"

    @property
    def requires_llm(self) -> bool:
        return True

    def compress(self, trajectory: Trajectory) -> Trajectory:
        if self.llm_client is None:
            raise ValueError("HierarchicalSummarizationStrategy requires LLM client")

        events = trajectory.events
        if len(events) <= self.config.max_events:
            return trajectory

        # 分段
        segments = self._segment_events(events)

        # 摘要每段
        summarized_events = []
        for segment in segments:
            summary_event = self._summarize_segment(segment, trajectory)
            summarized_events.append(summary_event)

        return Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=summarized_events,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            metadata={**trajectory.metadata, "compression": self.name, "original_length": len(events)}
        )

    def _segment_events(self, events: List[TrajectoryEvent]) -> List[List[TrajectoryEvent]]:
        """将事件分段"""
        segments = []
        for i in range(0, len(events), self.config.segment_size):
            segment = events[i:i + self.config.segment_size]
            segments.append(segment)
        return segments

    def _summarize_segment(self, segment: List[TrajectoryEvent], trajectory: Trajectory) -> TrajectoryEvent:
        """摘要一个段"""
        # 构建segment描述
        segment_text = self._format_segment(segment)

        prompt = self.config.summarization_prompt or SEGMENT_SUMMARIZATION_PROMPT
        prompt = prompt.format(
            segment=segment_text,
            task=trajectory.instruction[:500]
        )

        summary = self.llm_client.complete(prompt)

        # 创建摘要事件
        return TrajectoryEvent(
            index=-1,  # 摘要事件
            kind="SummaryEvent",
            action_type=self._determine_segment_type(segment),
            action={"kind": "Summary", "content": summary},
            action_kind="Summary",
            thought=[summary],
            raw_event={"type": "summary", "original_count": len(segment)}
        )

    def _format_segment(self, segment: List[TrajectoryEvent]) -> str:
        """格式化段落用于摘要"""
        lines = []
        for i, event in enumerate(segment):
            summary = event.get_action_summary(max_length=150)
            lines.append(f"{i+1}. [{event.action_type.value}] {summary}")
        return "\n".join(lines)

    def _determine_segment_type(self, segment: List[TrajectoryEvent]) -> ActionType:
        """确定段落的主要action类型"""
        type_counts = {}
        for event in segment:
            type_counts[event.action_type] = type_counts.get(event.action_type, 0) + 1

        if type_counts:
            return max(type_counts, key=type_counts.get)
        return ActionType.OTHER


SEGMENT_SUMMARIZATION_PROMPT = """Summarize this segment of a bug-fixing trajectory in 2-3 sentences.

Task: {task}

Segment actions:
{segment}

Focus on:
- What was discovered or accomplished
- Key files or functions examined/modified
- Important decisions made

Summary:"""
```

### 3. ActionTypeFilteringStrategy
```python
class ActionTypeFilteringStrategy(BaseCompressionStrategy):
    """
    Action类型过滤策略

    按action类型过滤事件
    """

    @property
    def name(self) -> str:
        return "action_type_filtering"

    def compress(self, trajectory: Trajectory) -> Trajectory:
        events = trajectory.events

        filtered = []
        prev_type = None

        for event in events:
            # 检查是否应该保留
            if not self._should_keep(event):
                continue

            # 去重连续相同类型
            if self.config.deduplicate_consecutive:
                if event.action_type == prev_type and event.action_type not in [ActionType.FILE_EDIT]:
                    continue

            filtered.append(event)
            prev_type = event.action_type

        # 限制数量
        if len(filtered) > self.config.max_events:
            filtered = self._limit_events(filtered)

        return Trajectory(
            instance_id=trajectory.instance_id,
            instruction=trajectory.instruction,
            events=filtered,
            git_patch=trajectory.git_patch,
            repository=trajectory.repository,
            metadata={**trajectory.metadata, "compression": self.name, "original_length": len(events)}
        )

    def _should_keep(self, event: TrajectoryEvent) -> bool:
        """判断是否保留事件"""
        # 排除列表优先
        if event.action_type in self.config.exclude_action_types:
            return False

        # 如果有保留列表，必须在列表中
        if self.config.keep_action_types:
            return event.action_type in self.config.keep_action_types

        return True

    def _limit_events(self, events: List[TrajectoryEvent]) -> List[TrajectoryEvent]:
        """限制事件数量，保留首尾"""
        if len(events) <= self.config.max_events:
            return events

        first_n = self.config.preserve_first_n
        last_n = self.config.preserve_last_n
        middle_count = self.config.max_events - first_n - last_n

        if middle_count <= 0:
            return events[:first_n] + events[-last_n:]

        # 从中间均匀采样
        middle = events[first_n:-last_n]
        step = len(middle) // middle_count
        sampled_middle = middle[::step][:middle_count]

        return events[:first_n] + sampled_middle + events[-last_n:]
```

---

## 主类: CompressionModule

```python
class CompressionModule:
    """
    Trajectory压缩模块

    Usage:
        # 单策略
        module = CompressionModule(strategy="key_step_extraction")
        compressed = module.compress(trajectory)

        # 组合策略
        module1 = CompressionModule(strategy="action_type_filtering")
        module2 = CompressionModule(strategy="key_step_extraction")
        combined = module1 + module2
        compressed = combined.compress(trajectory)
    """

    STRATEGY_MAP = {
        CompressionStrategy.KEY_STEP_EXTRACTION: KeyStepExtractionStrategy,
        CompressionStrategy.HIERARCHICAL_SUMMARIZATION: HierarchicalSummarizationStrategy,
        CompressionStrategy.ACTION_TYPE_FILTERING: ActionTypeFilteringStrategy,
    }

    def __init__(
        self,
        strategy: Union[CompressionStrategy, str] = CompressionStrategy.KEY_STEP_EXTRACTION,
        llm_client: Optional[LLMClient] = None,
        config: Optional[CompressionConfig] = None
    ):
        # 解析strategy
        if isinstance(strategy, str):
            strategy = CompressionStrategy(strategy)

        self.strategy_enum = strategy
        self.config = config or CompressionConfig()
        self.llm_client = llm_client

        # 创建策略实例
        strategy_class = self.STRATEGY_MAP[strategy]
        self._strategy = strategy_class(self.config, llm_client)

        # 检查LLM依赖
        if self._strategy.requires_llm and llm_client is None:
            raise ValueError(f"Strategy {strategy.value} requires LLM client")

    @property
    def strategy(self) -> str:
        """当前策略名称"""
        return self._strategy.name

    def compress(self, trajectory: Trajectory) -> Trajectory:
        """压缩单个trajectory"""
        return self._strategy.compress(trajectory)

    def compress_batch(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """批量压缩"""
        return self._strategy.compress_batch(trajectories)

    def __add__(self, other: "CompressionModule") -> "ComposedCompression":
        """组合两个压缩模块"""
        return ComposedCompression([self, other])

    def __repr__(self) -> str:
        return f"CompressionModule(strategy={self.strategy})"


class ComposedCompression:
    """
    组合压缩模块

    按顺序应用多个压缩策略
    """

    def __init__(self, modules: List[CompressionModule]):
        self.modules = modules

    def compress(self, trajectory: Trajectory) -> Trajectory:
        """依次应用所有压缩模块"""
        result = trajectory
        for module in self.modules:
            result = module.compress(result)
        return result

    def compress_batch(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        """批量压缩"""
        return [self.compress(t) for t in trajectories]

    def __add__(self, other: Union[CompressionModule, "ComposedCompression"]) -> "ComposedCompression":
        """继续组合"""
        if isinstance(other, CompressionModule):
            return ComposedCompression(self.modules + [other])
        else:
            return ComposedCompression(self.modules + other.modules)

    def __repr__(self) -> str:
        names = [m.strategy for m in self.modules]
        return f"ComposedCompression({' -> '.join(names)})"
```

---

## 依赖
- `CAWM/models.py`: Trajectory, TrajectoryEvent, ActionType
- `CAWM/llm_client.py`: LLMClient (HierarchicalSummarization策略需要)

## 测试要点
1. 各策略独立工作正常
2. KeyStepExtraction正确识别关键步骤
3. ActionTypeFiltering正确过滤
4. HierarchicalSummarization生成合理摘要
5. 组合压缩 `module1 + module2` 工作正常
6. 边界情况：空trajectory、短trajectory
