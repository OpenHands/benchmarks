# Stage 1-5: 归纳模块 (CAWM/induction.py)

## 目标
创建InductionModule，使用LLM从trajectory或cluster中归纳workflow，支持两级结构。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/induction.py`

---

## 枚举和配置

### WorkflowLevel (Enum)
```python
class WorkflowLevel(Enum):
    """Workflow层级"""
    GENERAL = 1   # 跨项目通用
    SPECIFIC = 2  # 项目/问题类型特定
```

### InductionConfig (dataclass)
```python
@dataclass
class InductionConfig:
    """归纳配置"""
    # 通用配置
    min_steps: int = 2                   # workflow最少步骤数
    max_steps: int = 8                   # workflow最多步骤数
    min_workflows: int = 1               # 最少提取workflow数
    max_workflows: int = 10              # 最多提取workflow数

    # LLM配置
    temperature: float = 0.0
    max_tokens: int = 4096

    # Level 1 (General) 配置
    general_abstraction_level: str = "high"  # high, medium, low
    general_use_placeholders: bool = True    # 使用占位符 {repo}, {file} 等

    # Level 2 (Specific) 配置
    specific_preserve_context: bool = True   # 保留项目/问题上下文
    specific_include_examples: bool = True   # 包含具体示例

    # 自定义prompt
    custom_system_prompt: Optional[str] = None
    custom_extraction_prompt: Optional[str] = None

    # 去重
    deduplicate: bool = True
    dedup_similarity_threshold: float = 0.8
```

---

## Prompt模板

```python
# Level 1: General Workflow Extraction
GENERAL_SYSTEM_PROMPT = """You are an expert at analyzing software engineering workflows.
Your task is to extract reusable, abstract workflow patterns from bug-fixing trajectories.
Focus on patterns that are applicable across different projects and repositories."""

GENERAL_EXTRACTION_PROMPT = """Analyze the following bug-fixing trajectories and extract {min_workflows}-{max_workflows} reusable workflow patterns.

## Requirements

1. **High Abstraction Level**:
   - Extract workflows that are applicable across different projects
   - Use placeholders: {{repo}}, {{file}}, {{function}}, {{test_file}}, {{pattern}}
   - Focus on the strategy, not specific implementation details

2. **Sub-routine Level**:
   - Each workflow should be 2-8 steps
   - Focus on a specific sub-task (e.g., "locate bug", "verify fix"), not the entire solution

3. **Categories**: Use one of:
   - exploration: Finding relevant code/files
   - investigation: Understanding the bug
   - modification: Making code changes
   - fix_and_verify: Applying fix and testing
   - testing: Running tests

## Trajectories

{trajectories}

## Output Format

For each workflow, use EXACTLY this format:

```
WORKFLOW: [Descriptive Name]
CATEGORY: [category]
DESCRIPTION: Use this workflow when [specific scenario]

STEP 1:
ENV: [Environment state before this step]
REASONING: [Why take this action]
ACTION: [Abstracted command with placeholders]
ACTION_TYPE: [exploration|file_view|file_edit|testing|terminal]

STEP 2:
...

---
```

Extract {min_workflows}-{max_workflows} workflows now:"""

# Level 2: Specific Workflow Extraction
SPECIFIC_SYSTEM_PROMPT = """You are an expert at analyzing software engineering workflows.
Your task is to extract specific, contextualized workflow patterns from bug-fixing trajectories.
Focus on patterns that capture project-specific or problem-type-specific knowledge."""

SPECIFIC_EXTRACTION_PROMPT = """Analyze the following bug-fixing trajectories and extract {min_workflows}-{max_workflows} specific workflow patterns.

## Context
These trajectories are from: {context}

## Requirements

1. **Preserve Context**:
   - Include project-specific patterns and conventions
   - Reference specific file structures or naming conventions when relevant
   - Capture problem-type-specific approaches

2. **Balance Abstraction**:
   - Use placeholders where appropriate, but preserve specific patterns
   - Include concrete examples when they illustrate important patterns

3. **Sub-routine Level**:
   - Each workflow should be 2-8 steps
   - Focus on specific sub-tasks relevant to this context

4. **Categories**: Use one of:
   - exploration, investigation, modification, fix_and_verify, testing

## Trajectories

{trajectories}

## Output Format

For each workflow:

```
WORKFLOW: [Name - can reference specific patterns]
CATEGORY: [category]
DESCRIPTION: Use this workflow when [specific scenario in this context]

STEP 1:
ENV: [State]
REASONING: [Why - can reference specific patterns]
ACTION: [Command - can include specific patterns]
ACTION_TYPE: [type]

STEP 2:
...

---
```

Extract {min_workflows}-{max_workflows} workflows now:"""
```

---

## Trajectory格式化

```python
class TrajectoryFormatter:
    """将Trajectory格式化为LLM可读的文本"""

    def __init__(self, max_events: int = 30, include_observation: bool = False):
        self.max_events = max_events
        self.include_observation = include_observation

    def format_single(self, trajectory: Trajectory) -> str:
        """格式化单个trajectory"""
        lines = [
            f"### Instance: {trajectory.instance_id}",
            f"Repository: {trajectory.repository}",
            f"Task: {trajectory.instruction[:500]}",
            "",
            "Actions taken:"
        ]

        events = trajectory.events[:self.max_events]
        for i, event in enumerate(events, 1):
            formatted = self._format_event(event)
            if formatted:
                lines.append(f"{i}. {formatted}")

        if len(trajectory.events) > self.max_events:
            lines.append(f"... ({len(trajectory.events) - self.max_events} more steps)")

        return "\n".join(lines)

    def format_batch(self, trajectories: List[Trajectory]) -> str:
        """格式化多个trajectory"""
        formatted = []
        for i, traj in enumerate(trajectories, 1):
            formatted.append(f"## Trajectory {i}")
            formatted.append(self.format_single(traj))
            formatted.append("\n" + "=" * 40 + "\n")

        return "\n".join(formatted)

    def format_cluster(self, cluster: TrajectoryCluster) -> str:
        """格式化cluster"""
        return self.format_batch(cluster.trajectories)

    def _format_event(self, event: TrajectoryEvent) -> Optional[str]:
        """格式化单个事件"""
        action_type = event.action_type.value

        if event.action_kind == "TerminalAction":
            cmd = event.command or ""
            if not cmd.strip():
                return None
            return f"[{action_type}] Terminal: {cmd[:150]}"

        elif event.action_kind == "FileEditorAction":
            cmd = event.action.get("command", "")
            path = event.path or ""

            if cmd == "view":
                return f"[file_view] View: {path}"
            elif cmd == "str_replace":
                old = event.action.get("old_str", "")[:50]
                new = event.action.get("new_str", "")[:50]
                return f"[file_edit] Edit {path}: {old}... -> {new}..."
            elif cmd == "create":
                return f"[file_edit] Create: {path}"
            else:
                return f"[{action_type}] {cmd}: {path}"

        elif event.action_kind == "ThinkAction":
            thought = event.thought[0] if event.thought else ""
            return f"[think] {thought[:100]}..."

        return None
```

---

## Workflow解析

```python
class WorkflowParser:
    """解析LLM输出为Workflow对象"""

    def __init__(self, config: InductionConfig):
        self.config = config

    def parse(self, response: str, level: WorkflowLevel, source_instances: List[str] = None) -> List[Workflow]:
        """解析LLM响应为Workflow列表"""
        from CAWM.llm_client import parse_workflow_blocks

        blocks = parse_workflow_blocks(response)
        workflows = []

        for block in blocks:
            workflow = self._block_to_workflow(block, level, source_instances)
            if workflow:
                workflows.append(workflow)

        # 去重
        if self.config.deduplicate:
            workflows = self._deduplicate(workflows)

        return workflows

    def _block_to_workflow(
        self,
        block: Dict,
        level: WorkflowLevel,
        source_instances: List[str]
    ) -> Optional[Workflow]:
        """将解析的block转换为Workflow对象"""
        name = block.get("name", "")
        category = block.get("category", "general")
        description = block.get("description", name)
        steps_data = block.get("steps", [])

        # 验证步骤数
        if len(steps_data) < self.config.min_steps:
            return None
        if len(steps_data) > self.config.max_steps:
            steps_data = steps_data[:self.config.max_steps]

        # 转换步骤
        steps = []
        for step_data in steps_data:
            step = WorkflowStep(
                env_description=step_data.get("env_description", ""),
                reasoning=step_data.get("reasoning", ""),
                action=step_data.get("action", ""),
                action_type=step_data.get("action_type", "terminal")
            )
            steps.append(step)

        # 生成ID
        workflow_id = self._generate_id(name, level)

        # 提取pattern
        pattern = tuple(s.action_type for s in steps)

        return Workflow(
            id=workflow_id,
            description=description,
            category=category,
            steps=steps,
            level=level.value,
            source_instances=source_instances or [],
            frequency=1,
            pattern=pattern
        )

    def _generate_id(self, name: str, level: WorkflowLevel) -> str:
        """生成workflow ID"""
        import hashlib
        content = f"{level.value}-{name}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        prefix = "gen" if level == WorkflowLevel.GENERAL else "spec"
        return f"wf-{prefix}-{hash_val}"

    def _deduplicate(self, workflows: List[Workflow]) -> List[Workflow]:
        """去重相似workflow"""
        if len(workflows) <= 1:
            return workflows

        unique = []
        seen_descriptions = []

        for wf in workflows:
            desc_lower = wf.description.lower().strip()

            # 检查是否与已有的相似
            is_duplicate = False
            for seen in seen_descriptions:
                if self._similarity(desc_lower, seen) > self.config.dedup_similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(wf)
                seen_descriptions.append(desc_lower)

        return unique

    def _similarity(self, s1: str, s2: str) -> float:
        """简单的字符串相似度"""
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
```

---

## 主类: InductionModule

```python
class InductionModule:
    """
    Workflow归纳模块

    使用LLM从trajectory或cluster中归纳workflow

    Usage:
        module = InductionModule(llm_client=client)

        # 从trajectories归纳
        workflows = module.induce(trajectories, level=WorkflowLevel.GENERAL)

        # 从clusters归纳
        workflows = module.induce_from_clusters(clusters, level=WorkflowLevel.SPECIFIC)

        # 两级归纳
        hierarchical = module.induce_hierarchical(trajectories)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[InductionConfig] = None
    ):
        if llm_client is None:
            raise ValueError("InductionModule requires LLM client")

        self.llm_client = llm_client
        self.config = config or InductionConfig()
        self.formatter = TrajectoryFormatter()
        self.parser = WorkflowParser(self.config)

    def induce(
        self,
        trajectories: List[Trajectory],
        level: WorkflowLevel = WorkflowLevel.GENERAL,
        context: str = ""
    ) -> List[Workflow]:
        """
        从trajectories归纳workflow

        Args:
            trajectories: Trajectory列表
            level: Workflow层级
            context: 上下文描述 (用于SPECIFIC level)

        Returns:
            Workflow列表
        """
        if not trajectories:
            return []

        # 格式化trajectories
        formatted = self.formatter.format_batch(trajectories)

        # 选择prompt
        if level == WorkflowLevel.GENERAL:
            system_prompt = self.config.custom_system_prompt or GENERAL_SYSTEM_PROMPT
            extraction_prompt = self.config.custom_extraction_prompt or GENERAL_EXTRACTION_PROMPT
        else:
            system_prompt = self.config.custom_system_prompt or SPECIFIC_SYSTEM_PROMPT
            extraction_prompt = self.config.custom_extraction_prompt or SPECIFIC_EXTRACTION_PROMPT

        # 构建prompt
        prompt = extraction_prompt.format(
            trajectories=formatted,
            min_workflows=self.config.min_workflows,
            max_workflows=self.config.max_workflows,
            context=context or self._infer_context(trajectories)
        )

        # 调用LLM
        response = self.llm_client.complete(
            prompt=prompt,
            system=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        # 解析结果
        source_instances = [t.instance_id for t in trajectories]
        workflows = self.parser.parse(response, level, source_instances)

        return workflows

    def induce_from_clusters(
        self,
        clusters: List[TrajectoryCluster],
        level: WorkflowLevel = WorkflowLevel.SPECIFIC
    ) -> List[Workflow]:
        """
        从clusters归纳workflow

        每个cluster产生一组workflow
        """
        all_workflows = []

        for cluster in clusters:
            context = f"Cluster {cluster.cluster_id} ({len(cluster)} trajectories)"

            workflows = self.induce(
                trajectories=cluster.trajectories,
                level=level,
                context=context
            )

            # 添加cluster信息到metadata
            for wf in workflows:
                wf.metadata["cluster_id"] = cluster.cluster_id

            all_workflows.extend(workflows)

        return all_workflows

    def induce_hierarchical(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[WorkflowLevel, List[Workflow]]:
        """
        两级归纳

        Returns:
            {
                WorkflowLevel.GENERAL: [...],
                WorkflowLevel.SPECIFIC: [...]
            }
        """
        result = {
            WorkflowLevel.GENERAL: [],
            WorkflowLevel.SPECIFIC: []
        }

        # Level 1: General workflows
        general_workflows = self.induce(
            trajectories=trajectories,
            level=WorkflowLevel.GENERAL
        )
        result[WorkflowLevel.GENERAL] = general_workflows

        # Level 2: Specific workflows (按repository分组)
        repo_groups = self._group_by_repository(trajectories)

        for repo, repo_trajectories in repo_groups.items():
            if len(repo_trajectories) >= 2:  # 至少2个才归纳
                specific_workflows = self.induce(
                    trajectories=repo_trajectories,
                    level=WorkflowLevel.SPECIFIC,
                    context=f"Repository: {repo}"
                )
                result[WorkflowLevel.SPECIFIC].extend(specific_workflows)

        return result

    def _infer_context(self, trajectories: List[Trajectory]) -> str:
        """从trajectories推断上下文"""
        repos = set(t.repository for t in trajectories if t.repository)
        if len(repos) == 1:
            return f"Repository: {list(repos)[0]}"
        elif repos:
            return f"Repositories: {', '.join(list(repos)[:3])}"
        return "Mixed trajectories"

    def _group_by_repository(self, trajectories: List[Trajectory]) -> Dict[str, List[Trajectory]]:
        """按repository分组"""
        groups = {}
        for traj in trajectories:
            repo = traj.repository or "unknown"
            if repo not in groups:
                groups[repo] = []
            groups[repo].append(traj)
        return groups

    def __repr__(self) -> str:
        return f"InductionModule(model={self.llm_client.model})"
```

---

## 依赖
- `CAWM/models.py`: Trajectory, TrajectoryCluster, Workflow, WorkflowStep
- `CAWM/llm_client.py`: LLMClient, parse_workflow_blocks

## 测试要点
1. General level归纳产生抽象workflow
2. Specific level归纳保留上下文
3. 两级归纳工作正常
4. 从cluster归纳正确关联cluster信息
5. 去重机制有效
6. 边界情况：空列表、单个trajectory
