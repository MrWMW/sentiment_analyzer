# sentiment_analyzer
made by Mr.W

基于BERT微调的改进总体话题情感分析

## 概述
本模型用于分析特定话题下所有评论的情感倾向，并根据话题整体情感倾向对个体评论的情感分析结果进行动态调整。模型基于预训练的BERT模型进行微调，特别适用于处理具有明显情感倾向的话题场景。

## 功能特点
- 批量分析CSV文件中的评论数据
- 计算话题整体情感基准值
- 根据话题倾向动态调整个体评论情感分类
- 生成详细的分类结果和统计报告
- 支持GPU加速处理

## 环境要求
  - Python 3.7+  
  - PyTorch 1.8+  
  - Transformers 4.0+  
  - pandas  
  - tqdm  
  - scipy  


## 安装依赖
`pip install torch transformers pandas tqdm scipy`

## 使用方式
- 注意：该仓库不包含模型文件，需自行下载或通过train.py进行训练。
- 训练代码：
`python train.py`

### 命令行参数
```
python sentiment_analyzer.py \
    --input <输入CSV文件路径> \ 
    [--model_path <模型路径>] \
    [--text_column <评论文本列名>] \
    [--output_dir <输出目录>] \
    [--max_length <最大序列长度>] \
    [--batch_size <批量大小>]
```
### 示例命令
```
python topic_sentiment_analysis.py \
    --input data/comments.csv \
    --model_path ./models/sentiment_model \
    --text_column content \
    --output_dir ./analysis_results \
    --batch_size 32
```
### 输入文件要求  
- CSV格式文件  
- 必须包含指定的评论文本列（默认为"评论"列）  
- 文本编码应为UTF-8  
- 文件应包含足够数量的评论（建议至少50条）以获得准确的话题情感基准  

### 输出文件说明  
- 处理完成后，将在输出目录生成以下文件：  
- <原文件名>_调整后积极评论.csv- 调整后的积极评论  
- <原文件名>_调整后消极评论.csv- 调整后的消极评论  
- <原文件名>_调整后中性评论.csv- 调整后的中性评论  
- <原文件名>_全部评论_processed2.csv- 包含所有分析结果的完整文件  
