# SkillEM

SkillEM 根据固定的成功执行轨迹推断可复用 skill，再联合训练 skill 生成与条件执行。工程参考本地 SearchLM-ExSkill，Search 环境、提示模板、动作投影和 EM 评分对齐已固定版本的 SkillRL。

```text
训练：问题 x + 成功轨迹 y → skill 后验 z → 联合 M-step
部署：问题 x → skill 先验 z → Search 环境交互 → 答案
```

目标为 `log pθ(z|x) + log pθ(y|x,z)`。这里的 `y` 是包含工具观察的成功执行轨迹，模型仅预测 assistant 动作。它与 ExSkill 的“隐搜索轨迹 + 答案似然加权”是不同的概率模型。

## 安装与本地验证

从本目录执行。CPU 冒烟测试不下载语言模型。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[train,test]'
pytest -q
bash scripts/smoke.sh
```

`materials/environment-lock.txt` 记录本地验证所用版本。虚拟环境、下载数据、模型权重和运行产物不随代码库发布。完整 GPU 实验使用 Linux/CUDA 环境；按服务器的 CUDA 版本安装 PyTorch。SkillRL/verl 和检索服务可使用独立环境，避免它们的 vLLM 依赖与本工程冲突。

`smoke.sh` 创建随机小 Transformer 和明确标记的合成轨迹，分别运行两轮 MCMC、Variational、SNIS、prior-uniform、SFT、static-skill，并检查恢复入口。它写入 `runs/smoke/`；重复运行请使用新的输出目录或移走旧冒烟结果。合成轨迹及 loss 不能作为 Search QA 效果数据。

## 材料与下载

SkillRL 固定于 `4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d`。`configs/assets.lock.json` 记录数据、模型、索引的仓库、commit、文件大小与发布的 SHA256；`materials/` 保存下载材料及收据。上游完整代码位于 `materials/SkillRL-<commit>/`。

```bash
python scripts/download_materials.py --group source
python scripts/download_materials.py --group data
python scripts/download_materials.py --group model
python scripts/download_materials.py --group retrieval --max-file-gb 50
```

官方 Hugging Face 域名连接失败时，可加 `--endpoint https://hf-mirror.com`；固定仓库版本与 LFS 校验值保持不变，下载收据记录实际地址。下载使用 `.part` 文件，完成并通过校验后才改为最终文件名。

Search SFT checkpoint 约 15.2 GB；E5 索引与压缩 Wikipedia 语料约 69.7 GB，解压和拼接还需额外空间。模型/检索文件的每文件下载上限默认 2 GiB，大索引需显式提高。`--metadata-only` 可仅下载模型配置和 tokenizer。

官方 `Jianwen/SkillRL-SFT-Data` 的 Search 部分有 1,214 条 instruction/output 单步样本，其中 500 条最终回答样本的 instruction 携带搜索历史和工具观察，可用于尝试重建轨迹。这 500 个问题均匹配训练集，未匹配测试集；按原始答案的官方严格 EM 检查，有 474 条最终回答通过。当前尚未完成逐步动作、观察和检索可重放性的审计，也未将它们转换为正式的成功轨迹训练集。`generated_memories_search.json` 是摘要经验，不能直接替代完整执行轨迹。

## 数据与 Search 协议

这里的 Search QA 指七项 Search-Augmented QA，不是 Jeopardy SearchQA 数据集。官方合并数据来自 `PeterJinGo/nq_hotpotqa_train`：训练为 NQ/HotpotQA；`test.parquet` 包含七项评测。精确版本及协议差异见 `configs/protocol.json`。

```bash
python -m skillem.cli prepare \
  --input materials/datasets/PeterJinGo--nq_hotpotqa_train/test.parquet \
  --output data/processed --repository PeterJinGo/nq_hotpotqa_train@b7d80abfee334a7a91cb377544f09180d58b34f6 \
  --split test
python -m skillem.cli prepare \
  --input materials/datasets/PeterJinGo--nq_hotpotqa_train/train.parquet \
  --output data/processed_clean --repository PeterJinGo/nq_hotpotqa_train@b7d80abfee334a7a91cb377544f09180d58b34f6 \
  --split train --seed 0 --exclude-questions data/processed/test.jsonl
python -m skillem.cli audit-data data/processed_clean/train.jsonl \
  data/processed_clean/validation.jsonl data/processed/test.jsonl
```

已确认发布数据中存在训练/测试重复问题。以上命令先保留完整测试集，再仅根据测试问题的规范化文本键排除训练重叠项；不使用测试答案或轨迹。本次共排除 10 条重叠记录，清理后为 161,194 条训练题、8,411 条验证题和完整的 51,713 条测试题。原始 parquet 保持不变，manifest 保存所有被排除的训练 ID，SkillRL 对照也应使用同一清理后的划分。

验证集按问题哈希从训练集划出，同一问题的重复实例留在同一集合。ID、原始来源、划分、文件哈希均保留。`audit-data` 检查跨集合重复问题并报错；它不会静默改写官方测试集。

每个任务记录 `task_id/question/answers/data_source/split/steps/verified/metadata`。每步包含原始 assistant `action`、工具 `observation` 和可选原始 `token_ids`。训练入口重新用官方评分验证成功状态，并拒绝测试数据、重复任务 ID 和空轨迹。多条成功轨迹应先按任务确定固定权重；当前参考入口每任务接收一条固定成功轨迹。

检索端点为 `POST /retrieve`，请求 `query/topk/return_scores`。默认 E5 + Wiki18、top-k=3、最多 4 个环境步、history=4、每步最多生成 700 token。准备完整检索材料后：

```bash
python scripts/prepare_retrieval.py
# 在检索环境中安装 torch/transformers/datasets、faiss-gpu、fastapi、uvicorn。
bash scripts/launch_retriever.sh
```

初次收集使用官方 Search SFT checkpoint，不注入尚未训练的生成式 skill：

```bash
python -m skillem.cli collect --config configs/search_mcmc.json \
  --input data/processed_clean/train.jsonl --output runs/collection --skill-mode none
```

保存全部成功与失败轨迹，`successes.jsonl` 才进入固定训练集。模型自由执行使用 `Task.public()`，屏蔽 gold answers 和已有成功轨迹。基础数据中的 `answers` 只交给环境评分，不参与部署 skill 排序。测试集不得用于收集训练成功轨迹。

## 训练与评测

```bash
python -m skillem.cli train --config configs/search_mcmc.json \
  --input runs/collection/successes.jsonl --output runs/search-mcmc \
  --validation data/processed_clean/validation.jsonl
python -m skillem.cli train --config configs/search_variational.json \
  --input runs/collection/successes.jsonl --output runs/search-variational \
  --validation data/processed_clean/validation.jsonl
```

配置中的学习率、EM 轮数、链数、预热与样本数是起始设置，不代表已调优。先用 `--limit 16` 检查服务器配置。`validation_limit` 固定每轮验证预算；验证按自由执行的 macro EM 选优，并支持 `patience`。输出 `best.json` 指向验证最优轮次，`latest.json` 指向最新完整轮次。

每轮保存后验样本、权重、链转移、诊断、模型/adapter、优化器与随机数状态。恢复使用相同配置、固定训练集和验证集，添加 `--resume`；轮内中断会从上一个完整轮次重做。`state.pt` 是本程序产生的本地检查点，只加载可信文件。

```bash
python -m skillem.cli train --config configs/search_mcmc.json \
  --input runs/collection/successes.jsonl --output runs/search-mcmc \
  --validation data/processed_clean/validation.jsonl --resume
# 把下方路径替换为 best.json 中记录的轮次。
python -m skillem.cli evaluate --config configs/search_mcmc.json \
  --checkpoint runs/search-mcmc/round-001/theta \
  --input data/processed/test.jsonl --output runs/search-mcmc-test --skill-mode prior
```

指标包含每个数据集的严格 EM、macro EM、平均步数、检索次数和生成 token 数。`episodes.jsonl` 保存失败在内的逐题记录。`--skill-mode drop` 与 `swap` 用于 skill 干预；`swap` 至少需要两个任务。部署不加载推断模型 φ。

## 后验与训练约定

五个字段固定为 `when_to_apply/preconditions/steps/verification/recovery`。每字段是有限长度的原始 token 序列：非 EOS 特殊 token 被 mask 后重新归一化；到长度上限时 EOS 的概率为 1。采样、先验评分、q 评分与编辑评分使用同一个定义。保留原始 IDs 避免 decode/re-encode 改变离散状态。该分布是本实现明确声明的有界生成模型，并非无条件复用原始 LM 的截断分数。

MCMC 支持独立 MH、局部字段编辑 MH、Metropolis-within-Gibbs，以及有限模型的精确 Gibbs。局部更新计算完整新状态的联合概率和正反向 proposal；随机 block 选择是固定均匀分布。混合核保留全支持独立通道和惰性项。预热后保留拒绝/停留状态；去重时用频率作为权重，M-step 不再乘执行似然。精确 Gibbs 只用于 `FiniteModel` 的完整可枚举字段空间。

Variational EM 使用独立的 θ/φ 参数实例和相同 token 支持。E-step 冻结 θ，使用 `log pθ(z|x)+log pθ(y|x,z)-log qφ(z|x,y)` 与 leave-one-out baseline；无组内标准差归一化、PPO clipping 或额外温度。更新 φ 后重新采样，再执行 M-step。surrogate loss 不作为 ELBO 报告。

M-step 对每个任务的后验期望等权平均，skill 与动作内部保留 token log-probability 的和。prompt、观察和 padding 不计入 loss。各分项独立反传以减少图的驻留，梯度累积按任务归一化。dropout 设为零；训练模式允许梯度检查点生效。超长上下文显式报错，不静默截断目标轨迹。

有限空间测试可检查精确 EM 单调性。大模型采用有限采样和 SGD，当前没有实现 proposal 的独立增益认证，也不声称逐轮真实似然单调。标量 R-hat/ESS 和接受率用于诊断，不能证明链已混合。

## 对照与已知边界

`configs/search_*.json` 提供 MCMC、Variational、SNIS、prior-uniform、SFT 和 static-skill。SNIS 是本 SkillEM 隐变量模型的先验 importance baseline；原始 ExSkill 代码保留在同级 `../SearchLM-ExSkill/`，二者实验名称应区分。

静态 skill 对照先运行 `scripts/attach_static_skills.py` 给训练/验证数据写入 `metadata.static_skill`，再用 `search_static_skill.json`。该检索器是词法 baseline，兼容两种 SkillBank 键名，不冒充官方 template/embedding 检索。

`scripts/prepare_skillrl_data.py` 可把相同任务划分导出为 SkillRL parquet；`scripts/run_skillrl_baseline.sh` 在上游独立环境运行官方 GRPO，并强制提供训练/验证文件和非零步数预算。上游 Search SkillBank 的 `query_type_skills` 与加载器读取的 `task_specific_skills` 不一致；比较原样运行与 schema 修正版时应分别命名并记录差异。wrapper 默认保留上游原样行为。`scripts/normalize_skill_bank.py` 可生成单独命名的修正版；已生成 `data/search_skills_schema_corrected.json`。使用修正版时，通过 Hydra 的 `env.skills_only_memory.skills_json_path` 参数显式指定它。

完整比较需要相同底座、成功轨迹、任务划分、adapter 容量及计算预算，并报告多随机种子。当前 HF 参考实现按序列执行，支持单设备全参数或 LoRA；没有实现 verl 分布式训练或 vLLM 高吞吐 rollout。本地验证不包括 7B GPU 训练、完整 Wiki18 检索和七项完整 QA 成绩。材料实际下载状态见 `materials/status.json`，验证记录见 `materials/verification.json`。

## 来源

- [SkillRL 官方仓库](https://github.com/aiming-lab/SkillRL/tree/4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d)
- [SkillRL 论文](https://arxiv.org/abs/2602.08234)
- 本地 `../SkillEM_Proposal/sections/proposal.tex`、`methods.tex` 和 `appendix.tex`
- 本地 `../SearchLM-ExSkill/` 的实现与分析

`skillem/vendor/` 的提示模板、动作投影和 EM 评分为固定上游文件，保留原版权头及 Apache-2.0 许可证，详见 `THIRD_PARTY.md`。
