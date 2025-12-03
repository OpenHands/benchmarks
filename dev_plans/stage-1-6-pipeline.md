# Stage 1-6: Pipeline编排器 (CAWM/pipeline.py)

## 目标
创建CAWMPipeline，编排Compression → Clustering → Induction的完整流程。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/pipeline.py`

---

## 配置

### PipelineConfig (dataclass)
```python
@dataclass
class PipelineConfig:
    """Pipeline配置"""
    # Compression配置
    compression_strategy: CompressionStrategy = CompressionStrategy.KEY_STEP_EXTRACTION
    compression_config: Optional[CompressionConfig] = None
    skip_compression: bool = False

    # Clustering配置
    clustering_method: SimilarityMethod = SimilarityMethod.PROBLEM_DESCRIPTION
    clustering_config: Optional[ClusteringConfig] = None
    skip_clustering: bool = False

    # Induction配置
    induction_config: Optional[InductionConfig] = None
    workflow_level: WorkflowLevel = WorkflowLevel.GENERAL
    hierarchical_induction: bool = False  # 两级归纳

    # 输出配置
    output_format: str = "json"           # json, jsonl
    save_intermediate: bool = False       # 保存中间结果
    intermediate_dir: Optional[str] = None

    # 日志
    verbose: bool = True
```

---

## 主类: CAWMPipeline

```python
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from CAWM.compression import CompressionModule, CompressionConfig, CompressionStrategy
from CAWM.clustering import ClusteringModule, ClusteringConfig, SimilarityMethod
from CAWM.induction import InductionModule, InductionConfig, WorkflowLevel
from CAWM.llm_client import LLMClient
from CAWM.models import Trajectory, TrajectoryCluster, Workflow


class CAWMPipeline:
    """
    CAWM完整Pipeline

    流程: Load → Compress → Cluster → Induce → Save

    Usage:
        # 完整pipeline
        pipeline = CAWMPipeline(llm_client=client)
        workflows = pipeline.run_from_file(
            "CAWM/trajectories/resolved_trajectories.jsonl",
            output_path="CAWM/workflow/new_workflows.json"
        )

        # 自定义配置
        config = PipelineConfig(
            compression_strategy=CompressionStrategy.ACTION_TYPE_FILTERING,
            workflow_level=WorkflowLevel.SPECIFIC
        )
        workflows = pipeline.run_from_file(input_path, config=config)

        # 分步执行
        trajectories = pipeline.load(input_path)
        compressed = pipeline.compress(trajectories)
        clusters = pipeline.cluster(compressed)
        workflows = pipeline.induce(clusters)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[PipelineConfig] = None
    ):
        self.llm_client = llm_client
        self.config = config or PipelineConfig()

        # 延迟初始化模块
        self._compression_module = None
        self._clustering_module = None
        self._induction_module = None

    # ==================== 模块初始化 ====================

    def _get_compression_module(self) -> CompressionModule:
        """获取压缩模块（延迟初始化）"""
        if self._compression_module is None:
            self._compression_module = CompressionModule(
                strategy=self.config.compression_strategy,
                llm_client=self.llm_client if self.config.compression_strategy == CompressionStrategy.HIERARCHICAL_SUMMARIZATION else None,
                config=self.config.compression_config
            )
        return self._compression_module

    def _get_clustering_module(self) -> ClusteringModule:
        """获取聚类模块（延迟初始化）"""
        if self._clustering_module is None:
            self._clustering_module = ClusteringModule(
                method=self.config.clustering_method,
                llm_client=self.llm_client,
                config=self.config.clustering_config
            )
        return self._clustering_module

    def _get_induction_module(self) -> InductionModule:
        """获取归纳模块（延迟初始化）"""
        if self._induction_module is None:
            self._induction_module = InductionModule(
                llm_client=self.llm_client,
                config=self.config.induction_config
            )
        return self._induction_module

    # ==================== 主要方法 ====================

    def run(
        self,
        trajectories: List[Trajectory],
        config: Optional[PipelineConfig] = None
    ) -> List[Workflow]:
        """
        运行完整pipeline

        Args:
            trajectories: Trajectory列表
            config: 覆盖默认配置

        Returns:
            提取的Workflow列表
        """
        if config:
            self.config = config

        self._log("Starting CAWM Pipeline")
        self._log(f"Input: {len(trajectories)} trajectories")

        # Step 1: Compression
        if not self.config.skip_compression:
            self._log(f"Step 1: Compression ({self.config.compression_strategy.value})")
            compressed = self.compress(trajectories)
            self._log(f"  Compressed to avg {self._avg_length(compressed):.1f} events per trajectory")
        else:
            self._log("Step 1: Compression (skipped)")
            compressed = trajectories

        # Save intermediate
        if self.config.save_intermediate:
            self._save_intermediate("compressed", compressed)

        # Step 2: Clustering
        if not self.config.skip_clustering:
            self._log(f"Step 2: Clustering ({self.config.clustering_method.value})")
            clusters = self.cluster(compressed)
            self._log(f"  Created {len(clusters)} clusters")
        else:
            self._log("Step 2: Clustering (skipped)")
            # 每个trajectory作为单独的cluster
            clusters = [
                TrajectoryCluster(
                    cluster_id=f"single_{i}",
                    trajectories=[t],
                    similarity_method="none"
                )
                for i, t in enumerate(compressed)
            ]

        # Save intermediate
        if self.config.save_intermediate:
            self._save_intermediate("clusters", clusters)

        # Step 3: Induction
        self._log(f"Step 3: Induction (level={self.config.workflow_level.name})")

        if self.config.hierarchical_induction:
            # 两级归纳
            hierarchical = self.induce_hierarchical(compressed)
            workflows = hierarchical[WorkflowLevel.GENERAL] + hierarchical[WorkflowLevel.SPECIFIC]
            self._log(f"  Extracted {len(hierarchical[WorkflowLevel.GENERAL])} general + {len(hierarchical[WorkflowLevel.SPECIFIC])} specific workflows")
        else:
            # 单级归纳
            workflows = self.induce(clusters)
            self._log(f"  Extracted {len(workflows)} workflows")

        self._log("Pipeline complete")
        return workflows

    def run_from_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None
    ) -> List[Workflow]:
        """
        从文件运行pipeline

        Args:
            input_path: 输入JSONL文件路径
            output_path: 输出JSON文件路径（可选）
            config: 覆盖默认配置

        Returns:
            提取的Workflow列表
        """
        # 加载
        trajectories = self.load(input_path)

        # 运行
        workflows = self.run(trajectories, config)

        # 保存
        if output_path:
            self.save(workflows, output_path)

        return workflows

    # ==================== 分步方法 ====================

    def load(self, file_path: str) -> List[Trajectory]:
        """加载trajectories"""
        self._log(f"Loading from {file_path}")
        return Trajectory.load_from_jsonl(file_path)

    def compress(
        self,
        trajectories: List[Trajectory],
        config: Optional[PipelineConfig] = None
    ) -> List[Trajectory]:
        """压缩trajectories"""
        if config:
            self.config = config

        module = self._get_compression_module()
        return module.compress_batch(trajectories)

    def cluster(
        self,
        trajectories: List[Trajectory],
        config: Optional[PipelineConfig] = None
    ) -> List[TrajectoryCluster]:
        """聚类trajectories"""
        if config:
            self.config = config

        module = self._get_clustering_module()
        return module.cluster(trajectories)

    def induce(
        self,
        clusters: List[TrajectoryCluster],
        config: Optional[PipelineConfig] = None
    ) -> List[Workflow]:
        """从clusters归纳workflows"""
        if config:
            self.config = config

        module = self._get_induction_module()
        return module.induce_from_clusters(clusters, level=self.config.workflow_level)

    def induce_hierarchical(
        self,
        trajectories: List[Trajectory],
        config: Optional[PipelineConfig] = None
    ) -> Dict[WorkflowLevel, List[Workflow]]:
        """两级归纳"""
        if config:
            self.config = config

        module = self._get_induction_module()
        return module.induce_hierarchical(trajectories)

    # ==================== I/O方法 ====================

    def save(self, workflows: List[Workflow], output_path: str) -> None:
        """保存workflows到文件"""
        self._log(f"Saving {len(workflows)} workflows to {output_path}")

        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 构建输出数据
        output_data = {
            "version": "2.0",
            "extraction_method": "cawm_pipeline",
            "created_at": datetime.now().isoformat(),
            "config": {
                "compression_strategy": self.config.compression_strategy.value,
                "clustering_method": self.config.clustering_method.value,
                "workflow_level": self.config.workflow_level.name,
                "hierarchical": self.config.hierarchical_induction
            },
            "llm_model": self.llm_client.model,
            "workflow_count": len(workflows),
            "workflows": [w.to_dict() for w in workflows]
        }

        # 保存
        if self.config.output_format == "jsonl":
            with open(output_path, "w") as f:
                for wf in workflows:
                    f.write(json.dumps(wf.to_dict()) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

    def _save_intermediate(self, name: str, data: Union[List[Trajectory], List[TrajectoryCluster]]) -> None:
        """保存中间结果"""
        if not self.config.intermediate_dir:
            return

        dir_path = Path(self.config.intermediate_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"{name}.jsonl"

        with open(file_path, "w") as f:
            for item in data:
                if isinstance(item, Trajectory):
                    # 简化保存
                    obj = {
                        "instance_id": item.instance_id,
                        "event_count": len(item.events),
                        "repository": item.repository
                    }
                else:  # TrajectoryCluster
                    obj = {
                        "cluster_id": item.cluster_id,
                        "size": len(item.trajectories),
                        "instances": [t.instance_id for t in item.trajectories]
                    }
                f.write(json.dumps(obj) + "\n")

        self._log(f"  Saved intermediate: {file_path}")

    # ==================== 辅助方法 ====================

    def _log(self, message: str) -> None:
        """日志输出"""
        if self.config.verbose:
            print(f"[CAWMPipeline] {message}")

    def _avg_length(self, trajectories: List[Trajectory]) -> float:
        """计算平均事件数"""
        if not trajectories:
            return 0.0
        return sum(len(t) for t in trajectories) / len(trajectories)

    def __repr__(self) -> str:
        return f"CAWMPipeline(model={self.llm_client.model})"
```

---

## 便捷函数

```python
def create_pipeline(
    provider: str = "openrouter",
    model: str = "anthropic/claude-3.5-sonnet",
    api_key: Optional[str] = None,
    **config_kwargs
) -> CAWMPipeline:
    """
    便捷函数：创建pipeline

    Usage:
        pipeline = create_pipeline(model="anthropic/claude-3.5-sonnet")
        workflows = pipeline.run_from_file("input.jsonl", "output.json")
    """
    llm_client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key
    )

    config = PipelineConfig(**config_kwargs) if config_kwargs else None

    return CAWMPipeline(llm_client=llm_client, config=config)


def run_pipeline(
    input_path: str,
    output_path: str,
    provider: str = "openrouter",
    model: str = "anthropic/claude-3.5-sonnet",
    api_key: Optional[str] = None,
    **config_kwargs
) -> List[Workflow]:
    """
    便捷函数：一步运行pipeline

    Usage:
        workflows = run_pipeline(
            "CAWM/trajectories/resolved_trajectories.jsonl",
            "CAWM/workflow/new_workflows.json",
            compression_strategy="key_step_extraction",
            workflow_level="GENERAL"
        )
    """
    pipeline = create_pipeline(
        provider=provider,
        model=model,
        api_key=api_key,
        **config_kwargs
    )

    return pipeline.run_from_file(input_path, output_path)
```

---

## CLI接口

```python
def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="CAWM Pipeline - Extract workflows from trajectories")

    parser.add_argument("input", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output JSON file path")

    # LLM配置
    parser.add_argument("--provider", default="openrouter", choices=["openrouter", "openai", "anthropic"])
    parser.add_argument("--model", default="anthropic/claude-3.5-sonnet")

    # Pipeline配置
    parser.add_argument("--compression", default="key_step_extraction",
                        choices=["key_step_extraction", "hierarchical_summarization", "action_type_filtering"])
    parser.add_argument("--clustering", default="problem_description",
                        choices=["problem_description", "action_sequence", "code_modification"])
    parser.add_argument("--level", default="GENERAL", choices=["GENERAL", "SPECIFIC"])
    parser.add_argument("--hierarchical", action="store_true", help="Enable two-level induction")

    # 跳过选项
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--skip-clustering", action="store_true")

    # 其他
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--intermediate-dir", default="./intermediate")
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    # 构建配置
    config = PipelineConfig(
        compression_strategy=CompressionStrategy(args.compression),
        clustering_method=SimilarityMethod(args.clustering),
        workflow_level=WorkflowLevel[args.level],
        hierarchical_induction=args.hierarchical,
        skip_compression=args.skip_compression,
        skip_clustering=args.skip_clustering,
        save_intermediate=args.save_intermediate,
        intermediate_dir=args.intermediate_dir,
        verbose=not args.quiet
    )

    # 运行
    pipeline = create_pipeline(
        provider=args.provider,
        model=args.model
    )
    pipeline.config = config

    output_path = args.output or args.input.replace(".jsonl", "_workflows.json")
    workflows = pipeline.run_from_file(args.input, output_path)

    print(f"\nExtracted {len(workflows)} workflows")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
```

---

## 依赖
- `CAWM/models.py`: Trajectory, TrajectoryCluster, Workflow
- `CAWM/llm_client.py`: LLMClient
- `CAWM/compression.py`: CompressionModule, CompressionConfig, CompressionStrategy
- `CAWM/clustering.py`: ClusteringModule, ClusteringConfig, SimilarityMethod
- `CAWM/induction.py`: InductionModule, InductionConfig, WorkflowLevel

## 测试要点
1. 完整pipeline运行正常
2. 分步方法（compress, cluster, induce）独立工作
3. 配置覆盖正确
4. 文件I/O正常
5. 跳过选项（skip_compression, skip_clustering）正确
6. CLI接口可用
