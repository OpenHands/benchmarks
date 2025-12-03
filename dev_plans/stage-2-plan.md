# Stage 2: 功能完善与增强 (Functionality Completion & Enhancement)

## 目标 (Objectives)
完善 CAWM (Code Agent Workflow Memory) 系统的核心功能，填补 Stage 1 留下的逻辑空白。主要聚焦于实现基于 LLM 的高级轨迹压缩、多维度的轨迹聚类策略，以及增强底层 LLM 客户端的鲁棒性。

## 任务分解 (Tasks)

### Task 2-1: LLM Client 鲁棒性增强 (Robustness)
**目标**: 解决网络波动、API 限流等问题，确保大规模处理时不中断。
**文件**: `CAWM/llm_client.py`
**待办事项**:
1.  **配置扩展**: 在 `__init__` 中添加 `retry_config` (如 `max_retries`, `backoff_factor`) 和 `timeout` 参数。
2.  **重试机制**: 实现指数退避 (Exponential Backoff) 重试逻辑，重点处理:
    *   Rate Limit Errors (HTTP 429)
    *   Server Errors (HTTP 5xx)
    *   Connection Timeouts
3.  **异常处理**: 统一封装底层 Provider (OpenAI, Anthropic) 抛出的特定异常。

### Task 2-2: 实现层次化摘要压缩 (Hierarchical Summarization)
**目标**: 利用 LLM 将冗长的轨迹步骤压缩为高层语义摘要，保留核心逻辑同时减少 Token 消耗。
**文件**: `CAWM/compression.py`
**方法**: `_compress_summarization`
**待办事项**:
1.  **分块策略 (Chunking)**: 将长轨迹按固定步数 (如 `chunk_size=10`) 或逻辑段落（如 ActionType 变化点）切分。
2.  **Prompt 设计**: 编写用于摘要的 System Prompt，要求 LLM 输入一系列 Action，输出一段简练的 Intent 描述。
3.  **摘要生成**: 循环调用 `llm_client` 生成摘要。
4.  **轨迹重构**: 将原始的 N 个 Event 替换为 1 个包含摘要信息的 `TrajectoryEvent` (ActionType 为 `THINK` 或 `SUMMARY`)，保留该片段内最重要的 Context (如修改的文件名)。

### Task 2-3: 实现高级聚类策略 (Advanced Clustering)
**目标**: 补全 `ClusteringModule` 中的占位方法，提供更多维度的相似性分析。
**文件**: `CAWM/clustering.py`

#### Subtask 2-3-1: 问题描述聚类 (`_cluster_problem_description`)
**待办事项**:
1.  **文本预处理**: 对 `instruction` 进行简单的分词和去停用词处理。
2.  **相似度计算**: 
    *   实现基于 **Jaccard Similarity** (词集合重叠度) 的基准算法。
    *   (可选优化) 如果效果不佳，利用 `llm_client` 提取 instruction 的 3-5 个关键标签 (Keywords)，再计算标签重叠度。
3.  **聚类算法**: 复用现有的贪心聚类逻辑，仅替换距离度量函数。

#### Subtask 2-3-2: 代码修改聚类 (`_cluster_code_modification`)
**待办事项**:
1.  **Patch 解析**: 使用项目依赖中的 `unidiff` 库解析 `trajectory.git_patch`。
2.  **特征提取**: 从 Patch 中提取：
    *   修改的文件路径列表 (File Paths)
    *   (进阶) 修改的函数/类名 (如果是 Python 代码)
3.  **相似度计算**: 计算两个轨迹修改文件集合的 Jaccard Similarity。修改相同文件的轨迹应被聚类在一起。

## 执行计划 (Execution Plan)

1.  **Step 1**: 完成 **Task 2-1 (LLM Client)**，为后续涉及 LLM 的操作打好基础。
2.  **Step 2**: 完成 **Task 2-3 (Clustering)**，因为这部分主要依赖逻辑算法，不强依赖 LLM 交互，便于快速验证。
3.  **Step 3**: 完成 **Task 2-2 (Compression)**，这部分逻辑最复杂且依赖 LLM，放在最后。
4.  **Step 4**: 编写 `tests/test_stage_2.py` 进行集成测试。

## 依赖检查
- `unidiff`: 用于 Patch 解析 (已在 `pyproject.toml` 中)。
- `tenacity`: 用于重试逻辑 (如果不在依赖中，需手动实现简单的 backoff 装饰器或添加依赖)。
    - *注*: 当前环境未显示 `tenacity`，将优先尝试手动实现轻量级重试。
