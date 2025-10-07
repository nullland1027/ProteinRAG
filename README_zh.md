# ProteinRAG

高速的蛋白质序列语义相似性检索系统，基于蛋白质语言模型嵌入（ESM2）与 Milvus Lite 向量数据库。相比传统基于序列比对的方式，采用嵌入向量语义检索可发现功能层面的相似性。

## ✨ 主要特性
- **ESM2 嵌入表示**（默认：`facebook/esm2_t6_8M_UR50D`，320维向量）
- **Milvus Lite 本地文件模式**（无需单独部署服务）
- **首次使用自动创建集合与索引**
- **Streamlit Web 界面**：上传 FASTA、查看概览、执行相似性检索
- **Top-K 相似结果检索**（1–20 可选）
- **安全清空功能**（需要 6 位动态验证码二次确认）
- **模型按需加载**（首次生成嵌入时加载，可主动预加载）
- **可复用 Python API**（`ProteinRAGService` 服务类）

## 🧱 架构概览
```
app.py (Streamlit 前端 UI)
 ├─ 使用 main.get_protein_service() -> 单例 ProteinRAGService
main.py
 ├─ ProteinRAGService
 │   ├─ connect_database() 连接 Milvus Lite
 │   ├─ create_collection_if_not_exists() 创建集合
 │   ├─ load_esm2_model() 加载 ESM2 模型
 │   ├─ process_fasta_file() 解析 FASTA + 生成嵌入
 │   ├─ insert_proteins() 插入向量与标量字段
 │   ├─ search_similar_proteins() 相似性检索 (L2)
 │   ├─ clear_database() 清空并重建集合
 │   └─ get_collection_stats() 获取统计
create_db.py （可选：独立初始化/校验工作流）
```

## 📂 项目结构
```
ProteinRAG/
├─ app.py                # Streamlit 应用入口
├─ main.py               # 核心服务（Milvus + ESM2 嵌入）
├─ create_db.py          # 可选的数据库初始化与验证脚本
├─ test_milvus_lite.py   # 基础连通性与集合测试
├─ requirements.txt      # 依赖列表
├─ milvus_lite.db        # Milvus Lite 本地存储文件（运行后生成）
├─ README.md             # 英文文档
└─ README_zh.md          # 中文文档
```

## 🛠 环境要求
- Python 3.10+（已在 3.12 测试）
- macOS / Linux（Windows 理论兼容，未充分验证）
- 足够内存（默认小模型，内存占用低）
- （可选）GPU（若切换到更大 ESM2 模型）

## 📦 安装步骤
```bash
# 1. 克隆仓库
git clone <your-fork-or-origin-url> ProteinRAG
cd ProteinRAG

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
```

## 🔐 环境变量 (.env)
如果需要自定义模型来源，可在项目根目录创建 `.env`：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `EMBEDDING_MODEL_PATH` | HuggingFace 模型名称或本地路径 | `facebook/esm2_t6_8M_UR50D` |

示例：
```
EMBEDDING_MODEL_PATH=facebook/esm2_t6_8M_UR50D
# 或本地路径
# EMBEDDING_MODEL_PATH=/data/models/esm2_t30_150M_UR50D
```
> 未设置时回退到轻量模型（加载快，适合开发验证）。

## ▶️ 启动 Web 应用
```bash
streamlit run app.py
```
浏览器访问（默认）：http://localhost:8501

### 界面使用流程
1. 在“Upload Protein Data”标签上传 FASTA 文件
2. 系统解析序列 → 生成 ESM2 嵌入 → 写入 Milvus Lite
3. 切换到“Protein Sequence Search”标签输入查询序列
4. 选择 K（1–20），执行搜索
5. 查看相似序列及相似度（1 / (1 + L2) 转换）

## 🧪 基础测试（Milvus Lite 连通性）
```bash
python test_milvus_lite.py
```
预期：所有步骤成功并输出统计。

## 🧬 编程调用示例
```python
from main import get_protein_service

service = get_protein_service()
service.initialize_database()

fasta_content = ">seq1\nMKT...\n>seq2\nGAVL..."
records = service.process_fasta_file(fasta_content)
service.insert_proteins(records)

results = service.search_similar_proteins("MKTVRQE...", top_k=5)
for r in results:
    print(r['protein_id'], r['similarity_score'])
```

## 📊 数据模式（Milvus 集合字段）
| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT64 (auto) | 主键，自增 |
| protein_id | VARCHAR(100) | FASTA 中的序列 ID |
| sequence | VARCHAR(10000) | 原始氨基酸序列（可加截断策略） |
| description | VARCHAR(1000) | FASTA 描述头 |
| length | INT64 | 序列长度 |
| embedding | FLOAT_VECTOR(320) | ESM2 CLS 位置嵌入 |

## 🧮 相似度度量
- 使用 **L2 距离** 作为 Milvus 计算基础
- 前端显示的相似度：`1 / (1 + distance)`（值越接近 1 越相似）

## 🗃 Milvus Lite 说明
- 采用本地文件，无需额外服务进程
- 首次自动生成 `milvus_lite.db`
- 索引类型使用 `AUTOINDEX`（Lite 受限支持）
- 若迁移到标准 Milvus 服务，可替换为 IVF / HNSW 等索引

## 🧹 清空数据库
- UI 侧“Clear Database”按钮触发
- 生成动态 6 位数字验证码，验证通过后删除并重建集合

## 🚀 性能建议
| 策略 | 效果 |
|------|------|
| 合并批量上传 | 降低模型调用开销 |
| 预加载模型 | 避免首请求延迟 |
| 使用更大模型 | 提升语义能力（牺牲速度） |
| 固定依赖版本 | 结果可复现 |

## 🧷 边界与行为
| 情况 | 行为 |
|------|------|
| 空 FASTA | 不写入，日志警告 |
| 非法字符 | 目前未过滤，可后续新增验证 |
| 重复 protein_id | 当前允许重复，可自定义去重策略 |
| >1000 aa 超长 | 预处理阶段截断到 1000（可修改） |
| 运行中删除 DB 文件 | 下次操作会尝试重建 |

## 🛠 开发流程建议
```bash
# 启动应用
streamlit run app.py

# 快速连通性测试
python test_milvus_lite.py
```
分支策略建议：
- main：稳定分支
- feat/<name>：新功能
- fix/<issue>：缺陷修复

## 🐞 常见问题排查
| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Failed to connect database | 路径或权限问题 | 检查写权限 / 删除锁文件 |
| 模型下载慢 | 网络 / 速率限制 | 预下载并设置本地路径 |
| 首次检索慢 | 模型懒加载 | 启动时调用 load_esm2_model |
| 结果为空 | 集合为空 | 先上传数据 |
| 清空失败 | 验证码错误 | 重新输入 |

## 🔄 可扩展方向
- 序列合法性过滤（20种氨基酸）
- 更大 ESM2 模型 + GPU 支持
- 余弦相似度选项
- 导出检索结果（CSV/JSON）
- FastAPI REST 接口
- 重复检测与合并

## 🔒 安全注意事项
- 当前无鉴权，仅适合本地/内网使用
- 上线需增加用户认证与访问控制
- 模型路径等敏感配置通过 `.env` 管理

## 📄 许可证
详见 [LICENSE](./LICENSE)

## 🙌 贡献
欢迎提交 Issue / PR：功能增强、性能优化、验证与清理。

---
**若在科研或生产中使用本项目，建议注明 ESM 与 Milvus 的原始项目。**

