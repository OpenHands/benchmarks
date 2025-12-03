# Stage 1-7: Package导出 (CAWM/__init__.py)

## 目标
更新CAWM包的`__init__.py`，导出所有公共接口。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/__init__.py`

---

## 内容

```python
"""
CAWM (Code Agent Workflow Memory)

A modular system for extracting reusable workflows from agent trajectories.

Modules:
- models: Core data models (Trajectory, Workflow, etc.)
- llm_client: OpenRouter-compatible LLM client
- compression: Trajectory compression strategies
- clustering: Trajectory clustering by similarity
- induction: LLM-based workflow induction
- pipeline: Complete CAWM pipeline orchestrator

Example Usage:
    from CAWM import CAWMPipeline, LLMClient, PipelineConfig

    # Create pipeline
    llm = LLMClient(provider="openrouter", model="anthropic/claude-3.5-sonnet")
    pipeline = CAWMPipeline(llm_client=llm)

    # Run extraction
    workflows = pipeline.run_from_file(
        "CAWM/trajectories/resolved_trajectories.jsonl",
        output_path="CAWM/workflow/new_workflows.json"
    )

    # Or use convenience function
    from CAWM import run_pipeline
    workflows = run_pipeline("input.jsonl", "output.json")
"""

# Version
__version__ = "2.0.0"

# ==================== Models ====================
from CAWM.models import (
    # Enums
    ActionType,

    # Data classes
    WorkflowStep,
    Workflow,
    TrajectoryEvent,
    Trajectory,
    TrajectoryCluster,

    # Utility functions
    classify_action_type,
    abstract_path,
    abstract_command,
)

# ==================== LLM Client ====================
from CAWM.llm_client import (
    LLMClient,
    LLMConfig,
    parse_workflow_blocks,
)

# ==================== Compression ====================
from CAWM.compression import (
    # Enum and Config
    CompressionStrategy,
    CompressionConfig,

    # Main class
    CompressionModule,
    ComposedCompression,

    # Strategy classes (for advanced usage)
    BaseCompressionStrategy,
    KeyStepExtractionStrategy,
    HierarchicalSummarizationStrategy,
    ActionTypeFilteringStrategy,
)

# ==================== Clustering ====================
from CAWM.clustering import (
    # Enum and Config
    SimilarityMethod,
    ClusteringConfig,

    # Main class
    ClusteringModule,

    # Calculator classes (for advanced usage)
    BaseSimilarityCalculator,
    ProblemDescriptionSimilarity,
    ActionSequenceSimilarity,
    CodeModificationSimilarity,
)

# ==================== Induction ====================
from CAWM.induction import (
    # Enum and Config
    WorkflowLevel,
    InductionConfig,

    # Main class
    InductionModule,

    # Helper classes
    TrajectoryFormatter,
    WorkflowParser,
)

# ==================== Pipeline ====================
from CAWM.pipeline import (
    # Config
    PipelineConfig,

    # Main class
    CAWMPipeline,

    # Convenience functions
    create_pipeline,
    run_pipeline,
)

# ==================== Public API ====================
__all__ = [
    # Version
    "__version__",

    # Models
    "ActionType",
    "WorkflowStep",
    "Workflow",
    "TrajectoryEvent",
    "Trajectory",
    "TrajectoryCluster",
    "classify_action_type",
    "abstract_path",
    "abstract_command",

    # LLM
    "LLMClient",
    "LLMConfig",
    "parse_workflow_blocks",

    # Compression
    "CompressionStrategy",
    "CompressionConfig",
    "CompressionModule",
    "ComposedCompression",

    # Clustering
    "SimilarityMethod",
    "ClusteringConfig",
    "ClusteringModule",

    # Induction
    "WorkflowLevel",
    "InductionConfig",
    "InductionModule",

    # Pipeline
    "PipelineConfig",
    "CAWMPipeline",
    "create_pipeline",
    "run_pipeline",
]
```

---

## 向后兼容

现有的`llm_base.py`和`rule_base.py`保持不变，它们可以：
1. 独立使用（现有功能）
2. 导入新模块的组件

可选：在`llm_base.py`中添加兼容导入：
```python
# llm_base.py (可选更新)
# 保持向后兼容，同时可以使用新组件
try:
    from CAWM.models import Workflow, WorkflowStep
    from CAWM.llm_client import LLMClient as NewLLMClient
except ImportError:
    pass  # 使用本地定义
```

---

## 测试

```python
# test_cawm_import.py
def test_imports():
    """测试所有导出是否可用"""
    from CAWM import (
        # Models
        ActionType, Workflow, WorkflowStep, Trajectory, TrajectoryEvent, TrajectoryCluster,

        # LLM
        LLMClient, LLMConfig,

        # Compression
        CompressionStrategy, CompressionConfig, CompressionModule,

        # Clustering
        SimilarityMethod, ClusteringConfig, ClusteringModule,

        # Induction
        WorkflowLevel, InductionConfig, InductionModule,

        # Pipeline
        PipelineConfig, CAWMPipeline, create_pipeline, run_pipeline
    )

    # 基本实例化测试
    assert ActionType.FILE_EDIT.value == "file_edit"
    assert WorkflowLevel.GENERAL.value == 1
    assert CompressionStrategy.KEY_STEP_EXTRACTION.value == "key_step_extraction"

    print("All imports successful!")

if __name__ == "__main__":
    test_imports()
```

---

## 依赖顺序

导入顺序很重要，因为模块之间有依赖关系：

1. `models.py` - 无依赖，基础数据类
2. `llm_client.py` - 无内部依赖
3. `compression.py` - 依赖 models, llm_client
4. `clustering.py` - 依赖 models, llm_client
5. `induction.py` - 依赖 models, llm_client
6. `pipeline.py` - 依赖所有其他模块

---

## 完成标准

- [ ] 所有公共类和函数都已导出
- [ ] `__all__` 列表完整
- [ ] 导入测试通过
- [ ] 文档字符串完整
- [ ] 版本号更新为2.0.0
