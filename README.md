## Language translation testplan (self-contained)

这份脚本尽量保留了你给的 `LanguageTranslationLLMConfig` / `LanguageTranslationServiceConfig` 类结构，但把项目内依赖（`src/...`、exporter、connector、metrics）补齐成**单仓库可运行**版本。

### 你需要提供的数据格式（和你代码一致）

用一个 CSV 文件（默认 `data/translation.csv`），至少包含以下列：

- **`question_id`**: int，每条样本唯一 id
- **`test_category`**: 固定写 `"language_translation"`
- **`test_id`**: 固定写 `"tran_text_en2es"`（你要跑 en→es）
- **`question`**: 英文原文（source）
- **`ground_truth_answer`**: 西语参考译文（reference）

示例（表头 + 两行）：

```csv
question_id,test_category,test_id,question,ground_truth_answer
1,language_translation,tran_text_en2es,"Hello, how are you?","Hola, ¿cómo estás?"
2,language_translation,tran_text_en2es,"Please translate this sentence into Spanish.","Por favor, traduce esta oración al español."
```

### 如何运行

1) 安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

2) 生成一份示例数据（可选）：

```bash
python3 language_translation_testplan.py --write_sample_data --data_path data/translation.csv
```

3) 用你 HuggingFace Hub 上的模型跑 en→es 推理并评测：

```bash
python3 language_translation_testplan.py \
  --data_path data/translation.csv \
  --test_id tran_text_en2es \
  --model_id YOUR_HF_ORG_OR_USER/YOUR_MODEL_NAME
```

输出：

- 控制台会打印平均指标：`bleu` / `rouge` / `meteor`（均为 0..1）
- 预测结果保存到：`outputs/language_translation/tran_text_en2es__predictions.csv`

### 注意（针对自定义模型）

- 这个 runner 默认用 `transformers` 的 `text2text-generation` pipeline 进行推理；这适用于大多数 Seq2Seq 翻译模型。
- 如果你的模型是 **LLaMA / Qwen / 其它 CausalLM 或 chat 模型**，脚本会自动切到 `text-generation`，并用一个翻译 prompt 来生成结果。
  - 需要按 chat 模板喂模型时，加 `--use_chat_template`
  - 需要自定义提示词时，加 `--prompt_template`（必须包含 `{source_lang}` `{target_lang}` `{text}`）
- `METEOR` 指标在某些环境里会依赖 NLTK 的额外语料（如 `wordnet`）；本脚本在缺少语料时会自动降级（不会让整个 test 失败）。

