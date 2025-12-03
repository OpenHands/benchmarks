# Stage 1-4: 聚类模块 (CAWM/clustering.py)

## 目标
创建ClusteringModule，支持三种相似性计算方法进行trajectory聚类。

## 文件路径
`/Users/tangyiq/dev/benchmarks/CAWM/clustering.py`

---

## 枚举和配置

### SimilarityMethod (Enum)
```python
class SimilarityMethod(Enum):
    """相似性计算方法"""
    PROBLEM_DESCRIPTION = "problem_description"
    ACTION_SEQUENCE = "action_sequence"
    CODE_MODIFICATION = "code_modification"
```

### ClusteringConfig (dataclass)
```python
@dataclass
class ClusteringConfig:
    """聚类配置"""
    # 通用配置
    num_clusters: Optional[int] = None   # 聚类数量，None则自动确定
    min_cluster_size: int = 2            # 最小聚类大小
    similarity_threshold: float = 0.5    # 相似度阈值
    linkage: str = "average"             # 层次聚类连接方式: single, complete, average

    # ProblemDescriptionSimilarity配置
    embedding_model: str = "text-embedding-3-small"
    use_tfidf_fallback: bool = True      # 无LLM时使用TF-IDF
    max_description_length: int = 1000   # 描述最大长度

    # ActionSequenceSimilarity配置
    ngram_range: Tuple[int, int] = (2, 4)  # N-gram范围
    action_type_weight: float = 0.7      # action类型权重
    action_detail_weight: float = 0.3    # action详情权重
    use_edit_distance: bool = True       # 使用编辑距离

    # CodeModificationSimilarity配置
    compare_file_paths: bool = True      # 比较修改的文件路径
    compare_file_types: bool = True      # 比较文件类型
    compare_change_patterns: bool = True # 比较修改模式
    compare_change_size: bool = True     # 比较修改规模
```

---

## 相似性计算基类

```python
from abc import ABC, abstractmethod

class BaseSimilarityCalculator(ABC):
    """相似性计算基类"""

    def __init__(self, config: ClusteringConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self._cache = {}  # 缓存计算结果

    @abstractmethod
    def calculate(self, traj1: Trajectory, traj2: Trajectory) -> float:
        """计算两个trajectory的相似度 (0-1)"""
        pass

    def calculate_matrix(self, trajectories: List[Trajectory]) -> np.ndarray:
        """计算相似度矩阵"""
        n = len(trajectories)
        matrix = np.zeros((n, n))

        for i in range(n):
            matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self.calculate(trajectories[i], trajectories[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    @property
    @abstractmethod
    def name(self) -> str:
        """方法名称"""
        pass

    @property
    def requires_llm(self) -> bool:
        """是否需要LLM"""
        return False
```

---

## 相似性实现

### 1. ProblemDescriptionSimilarity
```python
class ProblemDescriptionSimilarity(BaseSimilarityCalculator):
    """
    基于问题描述的语义相似性

    使用embedding计算issue description的相似度
    """

    @property
    def name(self) -> str:
        return "problem_description"

    @property
    def requires_llm(self) -> bool:
        return not self.config.use_tfidf_fallback

    def __init__(self, config: ClusteringConfig, llm_client: Optional[LLMClient] = None):
        super().__init__(config, llm_client)
        self._embeddings_cache = {}
        self._tfidf_vectorizer = None

    def calculate(self, traj1: Trajectory, traj2: Trajectory) -> float:
        # 获取描述
        desc1 = self._get_description(traj1)
        desc2 = self._get_description(traj2)

        # 获取embedding
        emb1 = self._get_embedding(desc1, traj1.instance_id)
        emb2 = self._get_embedding(desc2, traj2.instance_id)

        # 计算余弦相似度
        return self._cosine_similarity(emb1, emb2)

    def _get_description(self, trajectory: Trajectory) -> str:
        """获取问题描述"""
        desc = trajectory.instruction[:self.config.max_description_length]
        return desc

    def _get_embedding(self, text: str, cache_key: str) -> np.ndarray:
        """获取文本embedding"""
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]

        if self.llm_client:
            embedding = self._get_llm_embedding(text)
        else:
            embedding = self._get_tfidf_embedding(text)

        self._embeddings_cache[cache_key] = embedding
        return embedding

    def _get_llm_embedding(self, text: str) -> np.ndarray:
        """使用LLM获取embedding"""
        # 使用OpenAI embedding API
        import openai
        client = openai.OpenAI(api_key=self.llm_client.api_key)

        response = client.embeddings.create(
            model=self.config.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)

    def _get_tfidf_embedding(self, text: str) -> np.ndarray:
        """使用TF-IDF获取embedding (fallback)"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            # 需要先fit，这里简化处理

        # 简化实现：使用词频作为特征
        words = text.lower().split()
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # 转换为固定维度向量
        features = sorted(word_counts.keys())[:100]
        vector = [word_counts.get(f, 0) for f in features]

        # 填充到固定长度
        vector = vector + [0] * (100 - len(vector))
        return np.array(vector[:100], dtype=float)

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))
```

### 2. ActionSequenceSimilarity
```python
class ActionSequenceSimilarity(BaseSimilarityCalculator):
    """
    基于Action序列的相似性

    使用N-gram和编辑距离计算action序列相似度
    """

    @property
    def name(self) -> str:
        return "action_sequence"

    def calculate(self, traj1: Trajectory, traj2: Trajectory) -> float:
        # 获取action序列
        seq1 = self._get_action_sequence(traj1)
        seq2 = self._get_action_sequence(traj2)

        # 计算N-gram相似度
        ngram_sim = self._ngram_similarity(seq1, seq2)

        # 计算编辑距离相似度 (可选)
        if self.config.use_edit_distance:
            edit_sim = self._edit_distance_similarity(seq1, seq2)
            # 加权组合
            return 0.6 * ngram_sim + 0.4 * edit_sim

        return ngram_sim

    def _get_action_sequence(self, trajectory: Trajectory) -> List[str]:
        """获取action类型序列"""
        sequence = []
        for event in trajectory.events:
            action_type = event.action_type.value

            # 可选：加入action详情
            if self.config.action_detail_weight > 0:
                detail = self._get_action_detail(event)
                action_type = f"{action_type}:{detail}"

            sequence.append(action_type)

        return sequence

    def _get_action_detail(self, event: TrajectoryEvent) -> str:
        """获取action详情特征"""
        if event.action_kind == "FileEditorAction":
            # 返回操作类型: view, str_replace, create
            cmd = event.action.get("command", "")
            return cmd[:10]
        elif event.action_kind == "TerminalAction":
            # 返回命令前缀
            cmd = event.command or ""
            first_word = cmd.split()[0] if cmd.split() else ""
            return first_word[:10]
        return ""

    def _ngram_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """计算N-gram Jaccard相似度"""
        ngrams1 = set()
        ngrams2 = set()

        for n in range(self.config.ngram_range[0], self.config.ngram_range[1] + 1):
            ngrams1.update(self._extract_ngrams(seq1, n))
            ngrams2.update(self._extract_ngrams(seq2, n))

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _extract_ngrams(self, sequence: List[str], n: int) -> Set[Tuple[str, ...]]:
        """提取N-grams"""
        if len(sequence) < n:
            return set()

        ngrams = set()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngrams.add(ngram)

        return ngrams

    def _edit_distance_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """计算归一化编辑距离相似度"""
        m, n = len(seq1), len(seq2)

        if m == 0 and n == 0:
            return 1.0
        if m == 0 or n == 0:
            return 0.0

        # 动态规划计算编辑距离
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        edit_distance = dp[m][n]
        max_len = max(m, n)

        return 1.0 - (edit_distance / max_len)
```

### 3. CodeModificationSimilarity
```python
class CodeModificationSimilarity(BaseSimilarityCalculator):
    """
    基于代码修改的相似性

    比较git_patch中的修改模式
    """

    @property
    def name(self) -> str:
        return "code_modification"

    def calculate(self, traj1: Trajectory, traj2: Trajectory) -> float:
        patch1 = traj1.git_patch or ""
        patch2 = traj2.git_patch or ""

        if not patch1 and not patch2:
            return 1.0
        if not patch1 or not patch2:
            return 0.0

        similarities = []

        # 文件路径相似度
        if self.config.compare_file_paths:
            files1 = self._extract_file_paths(patch1)
            files2 = self._extract_file_paths(patch2)
            sim = self._jaccard_similarity(files1, files2)
            similarities.append(sim)

        # 文件类型相似度
        if self.config.compare_file_types:
            types1 = self._extract_file_types(patch1)
            types2 = self._extract_file_types(patch2)
            sim = self._jaccard_similarity(types1, types2)
            similarities.append(sim)

        # 修改模式相似度
        if self.config.compare_change_patterns:
            patterns1 = self._extract_change_patterns(patch1)
            patterns2 = self._extract_change_patterns(patch2)
            sim = self._jaccard_similarity(patterns1, patterns2)
            similarities.append(sim)

        # 修改规模相似度
        if self.config.compare_change_size:
            size1 = self._get_change_size(patch1)
            size2 = self._get_change_size(patch2)
            max_size = max(size1, size2)
            sim = 1.0 - abs(size1 - size2) / max_size if max_size > 0 else 1.0
            similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _extract_file_paths(self, patch: str) -> Set[str]:
        """提取修改的文件路径"""
        import re
        # 匹配 diff --git a/path b/path 或 +++ b/path
        patterns = [
            r'diff --git a/(.+?) b/',
            r'\+\+\+ b/(.+)',
            r'--- a/(.+)'
        ]

        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, patch)
            files.update(matches)

        return files

    def _extract_file_types(self, patch: str) -> Set[str]:
        """提取文件类型"""
        files = self._extract_file_paths(patch)
        types = set()

        for f in files:
            ext = f.split('.')[-1] if '.' in f else 'unknown'
            types.add(ext)

        return types

    def _extract_change_patterns(self, patch: str) -> Set[str]:
        """提取修改模式特征"""
        patterns = set()

        # 检测常见模式
        pattern_checks = [
            (r'def \w+', 'function_definition'),
            (r'class \w+', 'class_definition'),
            (r'import ', 'import_statement'),
            (r'raise \w+', 'exception_handling'),
            (r'try:', 'try_block'),
            (r'except', 'except_block'),
            (r'return ', 'return_statement'),
            (r'if .+:', 'conditional'),
            (r'for .+ in', 'loop'),
            (r'while .+:', 'while_loop'),
            (r'assert ', 'assertion'),
            (r'@\w+', 'decorator'),
        ]

        for regex, pattern_name in pattern_checks:
            if re.search(regex, patch):
                patterns.add(pattern_name)

        return patterns

    def _get_change_size(self, patch: str) -> int:
        """获取修改规模 (添加+删除的行数)"""
        additions = len(re.findall(r'^\+[^+]', patch, re.MULTILINE))
        deletions = len(re.findall(r'^-[^-]', patch, re.MULTILINE))
        return additions + deletions

    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Jaccard相似度"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0
```

---

## 主类: ClusteringModule

```python
class ClusteringModule:
    """
    Trajectory聚类模块

    Usage:
        module = ClusteringModule(method="problem_description")
        clusters = module.cluster(trajectories)

        # 或计算相似度
        sim = module.get_similarity(traj1, traj2)
    """

    CALCULATOR_MAP = {
        SimilarityMethod.PROBLEM_DESCRIPTION: ProblemDescriptionSimilarity,
        SimilarityMethod.ACTION_SEQUENCE: ActionSequenceSimilarity,
        SimilarityMethod.CODE_MODIFICATION: CodeModificationSimilarity,
    }

    def __init__(
        self,
        method: Union[SimilarityMethod, str] = SimilarityMethod.PROBLEM_DESCRIPTION,
        llm_client: Optional[LLMClient] = None,
        config: Optional[ClusteringConfig] = None
    ):
        # 解析method
        if isinstance(method, str):
            method = SimilarityMethod(method)

        self.method_enum = method
        self.config = config or ClusteringConfig()
        self.llm_client = llm_client

        # 创建相似性计算器
        calculator_class = self.CALCULATOR_MAP[method]
        self._calculator = calculator_class(self.config, llm_client)

        # 检查LLM依赖
        if self._calculator.requires_llm and llm_client is None:
            if not self.config.use_tfidf_fallback:
                raise ValueError(f"Method {method.value} requires LLM client")

    @property
    def method(self) -> str:
        """当前方法名称"""
        return self._calculator.name

    def get_similarity(self, traj1: Trajectory, traj2: Trajectory) -> float:
        """计算两个trajectory的相似度"""
        return self._calculator.calculate(traj1, traj2)

    def cluster(self, trajectories: List[Trajectory]) -> List[TrajectoryCluster]:
        """聚类trajectories"""
        if len(trajectories) < 2:
            # 单个trajectory作为一个cluster
            if trajectories:
                return [TrajectoryCluster(
                    cluster_id="cluster_0",
                    trajectories=trajectories,
                    similarity_method=self.method
                )]
            return []

        # 计算相似度矩阵
        sim_matrix = self._calculator.calculate_matrix(trajectories)

        # 转换为距离矩阵
        dist_matrix = 1 - sim_matrix

        # 层次聚类
        clusters = self._hierarchical_clustering(trajectories, dist_matrix)

        return clusters

    def _hierarchical_clustering(
        self,
        trajectories: List[Trajectory],
        dist_matrix: np.ndarray
    ) -> List[TrajectoryCluster]:
        """层次聚类"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # 转换为压缩距离矩阵
        condensed = squareform(dist_matrix)

        # 层次聚类
        Z = linkage(condensed, method=self.config.linkage)

        # 确定聚类数量
        if self.config.num_clusters:
            n_clusters = self.config.num_clusters
            labels = fcluster(Z, n_clusters, criterion='maxclust')
        else:
            # 使用距离阈值
            threshold = 1 - self.config.similarity_threshold
            labels = fcluster(Z, threshold, criterion='distance')

        # 组建clusters
        cluster_dict = {}
        for i, label in enumerate(labels):
            if label not in cluster_dict:
                cluster_dict[label] = []
            cluster_dict[label].append(trajectories[i])

        # 过滤小cluster
        clusters = []
        for label, trajs in cluster_dict.items():
            if len(trajs) >= self.config.min_cluster_size:
                cluster = TrajectoryCluster(
                    cluster_id=f"cluster_{label}",
                    trajectories=trajs,
                    similarity_method=self.method
                )
                clusters.append(cluster)

        return clusters

    def __repr__(self) -> str:
        return f"ClusteringModule(method={self.method})"
```

---

## 依赖
- `CAWM/models.py`: Trajectory, TrajectoryCluster, ActionType
- `CAWM/llm_client.py`: LLMClient (ProblemDescriptionSimilarity可选)
- `numpy`: 矩阵运算
- `scipy`: 层次聚类
- `sklearn`: TF-IDF fallback (可选)

## 测试要点
1. 各相似性方法计算正确
2. 相似度矩阵对称且对角线为1
3. 聚类结果合理
4. 无LLM时TF-IDF fallback正常
5. 边界情况：空列表、单个trajectory
