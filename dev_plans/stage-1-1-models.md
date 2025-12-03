# Stage 1-1: 数据模型 (CAWM/models.py)

## 目标
创建CAWM系统的核心数据模型，作为所有模块的基础。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/models.py`

---

## 数据类设计

### 1. ActionType (Enum)
```python
class ActionType(Enum):
    """Agent action类型分类"""
    EXPLORATION = "exploration"      # 搜索、grep、find等
    FILE_VIEW = "file_view"          # 查看文件内容
    FILE_EDIT = "file_edit"          # 编辑文件
    TESTING = "testing"              # 运行测试
    TERMINAL = "terminal"            # 通用终端命令
    NAVIGATION = "navigation"        # cd、pwd等导航
    VERSION_CONTROL = "version_control"  # git操作
    SETUP = "setup"                  # pip、conda等环境设置
    THINK = "think"                  # 思考/推理
    OTHER = "other"                  # 其他
```

### 2. WorkflowStep (dataclass)
保持与现有`llm_base.py`兼容：
```python
@dataclass
class WorkflowStep:
    """Workflow中的单个步骤 - 保持现有格式"""
    env_description: str    # 执行前的环境状态描述
    reasoning: str          # 为什么执行这个action
    action: str             # 抽象化的action命令
    action_type: str        # action类型字符串

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep": ...
```

### 3. Workflow (dataclass)
扩展现有格式，增加level字段：
```python
@dataclass
class Workflow:
    """可复用的workflow"""
    id: str                              # 唯一标识符
    description: str                     # 使用场景描述
    category: str                        # 类别: exploration, investigation, modification, fix_and_verify, testing
    steps: List[WorkflowStep]            # 步骤列表
    level: int = 1                       # 1=通用, 2=特定
    source_instances: List[str] = field(default_factory=list)  # 来源instance IDs
    frequency: int = 1                   # 出现频率
    pattern: Tuple[str, ...] = field(default_factory=tuple)    # action类型模式
    metadata: Dict[str, Any] = field(default_factory=dict)     # 额外元数据

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow": ...
```

### 4. TrajectoryEvent (dataclass)
解析后的trajectory事件：
```python
@dataclass
class TrajectoryEvent:
    """解析后的trajectory事件"""
    index: int                           # 事件序号
    kind: str                            # 事件类型: ActionEvent, ObservationEvent
    action_type: ActionType              # 分类后的action类型
    action: Dict[str, Any]               # 原始action数据
    action_kind: str                     # action的kind: TerminalAction, FileEditorAction, ThinkAction
    thought: List[str]                   # thinking内容列表
    command: Optional[str] = None        # 执行的命令（如果有）
    path: Optional[str] = None           # 操作的文件路径（如果有）
    observation: Optional[Dict[str, Any]] = None  # observation数据
    observation_content: Optional[str] = None     # observation文本内容
    tool_call_id: Optional[str] = None   # tool call ID
    raw_event: Dict[str, Any] = field(default_factory=dict)  # 原始事件数据

    @classmethod
    def from_raw_event(cls, event: Dict[str, Any], index: int) -> Optional["TrajectoryEvent"]:
        """从原始事件解析"""
        ...

    def get_action_summary(self, max_length: int = 200) -> str:
        """获取action摘要"""
        ...

    def is_key_step(self) -> bool:
        """判断是否为关键步骤"""
        ...
```

### 5. Trajectory (dataclass)
完整的trajectory包装：
```python
@dataclass
class Trajectory:
    """完整的trajectory"""
    instance_id: str                     # instance唯一标识
    instruction: str                     # 任务指令/问题描述
    events: List[TrajectoryEvent]        # 解析后的事件列表
    git_patch: Optional[str] = None      # 生成的git patch
    repository: str = ""                 # 仓库名 (从instance_id解析)
    issue_type: str = ""                 # 问题类型 (可选)
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "Trajectory":
        """从原始JSONL数据解析"""
        ...

    @classmethod
    def load_from_jsonl(cls, file_path: str) -> List["Trajectory"]:
        """从JSONL文件加载多个trajectory"""
        ...

    def get_action_sequence(self) -> List[ActionType]:
        """获取action类型序列"""
        ...

    def get_key_events(self) -> List[TrajectoryEvent]:
        """获取关键事件"""
        ...

    def __len__(self) -> int:
        return len(self.events)
```

### 6. TrajectoryCluster (dataclass)
聚类结果：
```python
@dataclass
class TrajectoryCluster:
    """Trajectory聚类"""
    cluster_id: str                      # 聚类ID
    trajectories: List[Trajectory]       # 聚类中的trajectory列表
    label: str = ""                      # 聚类标签/描述
    similarity_method: str = ""          # 使用的相似性方法
    centroid: Optional[Any] = None       # 聚类中心（可选）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trajectories)

    def get_instance_ids(self) -> List[str]:
        """获取所有instance IDs"""
        ...
```

---

## 辅助函数

### Action类型分类
```python
def classify_action_type(event: Dict[str, Any]) -> ActionType:
    """根据事件内容分类action类型"""
    # 使用正则匹配命令模式
    EXPLORATION_PATTERNS = [r'\bfind\b', r'\bgrep\b', r'\brg\b', r'\bls\b', r'\bcat\b', ...]
    TESTING_PATTERNS = [r'\bpytest\b', r'\bpython.*test', r'\btox\b', ...]
    # ... 其他模式
```

### 路径抽象化
```python
def abstract_path(path: str, repo_name: str = "") -> str:
    """将具体路径抽象为模板"""
    # /workspace/django/models.py -> {repo}/models.py
    ...

def abstract_command(command: str, repo_name: str = "") -> str:
    """将具体命令抽象为模板"""
    # grep -r "MyClass" /workspace/django -> grep -r "{pattern}" {repo}
    ...
```

---

## 依赖
- Python 标准库: `dataclasses`, `enum`, `typing`, `json`, `re`
- 无外部依赖

## 测试要点
1. `Trajectory.from_raw()` 能正确解析 `resolved_trajectories.jsonl` 中的数据
2. `TrajectoryEvent.from_raw_event()` 能处理各种action类型
3. `classify_action_type()` 分类准确
4. `to_dict()` 和 `from_dict()` 双向转换正确
