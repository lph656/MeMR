# Reviewer Response Draft

感谢审稿人的提醒。我们已补充一个独立的数据完整性审计，覆盖 CMedCL 的源文件切分规则、内部重复检查、以及与当前可获得的通用医疗评测集之间的重叠检查。

根据当前仓库快照可以明确说明：

- CMedCL 由 `datasets/medical_consult/{neike, waike, erke, fuchanke, nanke, zhongliuke}/train.json` 与 `test.json` 组成。
- 训练时的验证集来自各科室 `train.json` 的确定性 10% 切分，代码中使用 `seed=0`。
- 我们已生成内部重复/跨切分审计表，以及与 `ChatMed_Consult-v0.3_test_500`、`Chinese-medical-dialogue-data_test_500`、`huatuo26M_test_500` 的精确匹配与近重复检查结果。

当前仓库快照中仍未能恢复的项目包括：完整清洗 prompt、清洗时的具体 LLM 版本与 temperature、人工复核流程、以及数据集级别的 release license。这些项目已在审计报告中显式标记为未恢复项，而不是推断填充。

请将 `prompt_r3_c7_report.md` 中的统计值插入正文即可形成最终 rebuttal 版本。

