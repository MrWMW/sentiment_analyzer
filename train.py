# -*- coding: utf-8 -*-

# ==============================================================================
# 1. 导入所需库
# ==============================================================================
import os
import torch
import transformers
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# ==============================================================================
# 2. 定义全局常量
# ==============================================================================
MODEL_NAME = 'hfl/chinese-bert-wwm'
DATASET_NAME = 'ChnSentiCorp.csv'
POS_FILE = 'pos60000.txt'  # 积极情绪数据文件
NEG_FILE = 'neg60000.txt'  # 消极情绪数据文件
OUTPUT_DIR = './output/bert-chnsenticorp-finetuned'
MAX_LENGTH = 128
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5


# ==============================================================================
# 3. 定义功能函数
# ==============================================================================
def setup_environment():
    """设置运行环境并检查GPU可用性"""
    transformers.logging.set_verbosity_info()
    
    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
    
    return device


def load_and_preprocess_data(tokenizer):
    """加载和预处理数据集"""
    print("\n--- 步骤 1: 加载并处理数据集 ---")
    
    # 加载ChnSentiCorp.csv数据集
    print(f"正在从 '{DATASET_NAME}' 加载数据集...")
    dataset = load_dataset('csv', data_files=DATASET_NAME)
    
    # 重命名列以符合标准格式
    dataset = dataset.rename_column("review", "text")
    dataset = dataset.rename_column("label", "labels")
    
    # 加载额外的积极和消极情绪数据
    print("正在加载额外数据: pos60000.txt 和 neg60000.txt")
    
    def load_text_file(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        texts = [line.strip() for line in lines]
        labels = [label] * len(texts)
        return {'text': texts, 'labels': labels}
    
    # 加载积极数据
    pos_data = load_text_file(POS_FILE, 1)
    pos_dataset = Dataset.from_dict(pos_data)
    
    # 加载消极数据
    neg_data = load_text_file(NEG_FILE, 0)
    neg_dataset = Dataset.from_dict(neg_data)
    
    # 合并数据集
    combined_dataset = concatenate_datasets([dataset['train'], pos_dataset, neg_dataset])
    
    # 分割数据集
    train_testvalid = combined_dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    train_dataset = train_testvalid['train']
    eval_dataset = test_valid['train']
    test_dataset = test_valid['test']
    
    # 检查数据集中的异常值
    print("检查数据集中文本列的数据类型...")
    
    # 过滤掉非文本数据
    def filter_non_text(example):
        return isinstance(example['text'], str)
    
    train_dataset = train_dataset.filter(filter_non_text)
    eval_dataset = eval_dataset.filter(filter_non_text)
    test_dataset = test_dataset.filter(filter_non_text)
    
    print(f"过滤后训练集大小: {len(train_dataset)}")
    
    # 定义数据处理函数
    def tokenize_function(examples):
        """对文本进行分词、编码"""
        texts = examples['text']
        texts = [str(text) if text is not None else "" for text in texts]
        
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=MAX_LENGTH
        )
    
    # 使用 .map() 方法批量处理数据
    print("\n正在对数据集进行预处理...")
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=1000,  # 增加批处理大小
        load_from_cache_file=False  # 禁用缓存以避免IO瓶颈
    )
    eval_dataset = eval_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=1000,
        load_from_cache_file=False
    )
    test_dataset = test_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=1000,
        load_from_cache_file=False
    )
    
    # 设置格式为 PyTorch 张量
    columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in train_dataset.features:
        columns.append('token_type_ids')
        
    train_dataset.set_format(type='torch', columns=columns)
    eval_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)
    
    print("数据集处理完成！")
    
    return train_dataset, eval_dataset, test_dataset


def load_model(device):
    """加载预训练模型"""
    print("\n--- 步骤 2: 加载预训练模型 ---")
    print(f"正在加载 '{MODEL_NAME}' 模型用于序列分类...")
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # 将模型移动到 GPU（如果可用）
    model.to(device)
    print(f"模型已移动到: {next(model.parameters()).device}")
    
    print("模型加载完成！")
    return model


def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


def setup_trainer(model, train_dataset, eval_dataset):
    """设置训练器和训练参数"""
    print("\n--- 步骤 3: 设置训练参数 ---")
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",  # 改为steps而不是epoch
        eval_steps=500,  # 每500步评估一次
        save_strategy="steps",
        save_steps=500,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=32,  # 增加批次大小
        per_device_eval_batch_size=32,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=False,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # 增加工作进程数
        save_total_limit=2,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        optim="adamw_torch",  # 使用融合优化器
        report_to="none",  # 禁用默认的报告以减少开销
    )
    
    # 实例化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    return trainer


def evaluate_model(trainer, test_dataset):
    """评估模型性能"""
    print("\n--- 步骤 4: 在测试集上评估模型 ---")
    test_results = trainer.evaluate(test_dataset)
    print("测试集评估结果:")
    print(test_results)
    return test_results


def save_model(model, tokenizer):
    """保存模型和分词器"""
    print(f"\n正在将最终模型保存到 '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("模型保存完毕。")


def inference_example(model_path, tokenizer_path, sample_texts):
    """进行推理示例"""
    print("\n--- 步骤 5: 进行推理验证 ---")
    
    # 加载微调后的模型进行推理
    print("加载微调后的模型进行推理...")
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 标签映射
    id2label = {0: "差评 (Negative)", 1: "好评 (Positive)"}
    
    # 准备输入
    inputs = loaded_tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"  # 返回PyTorch Tensors
    )
    
    # 模型推理
    outputs = loaded_model(**inputs)
    logits = outputs.logits
    predictions = np.argmax(logits.detach().numpy(), axis=1)
    
    print("\n--- 推理结果 ---")
    for text, pred in zip(sample_texts, predictions):
        sentiment = id2label[pred]
        print(f"评论: {text}")
        print(f"  -> 预测情感: {sentiment}\n")
    
    return predictions


# ==============================================================================
# 4. 主程序
# ==============================================================================
def main():
    """主函数"""
    # 设置环境
    device = setup_environment()
    
    # 加载分词器
    print(f"\n正在从 '{MODEL_NAME}' 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 加载和预处理数据
    train_dataset, eval_dataset, test_dataset = load_and_preprocess_data(tokenizer)
    
    # 加载模型
    model = load_model(device)
    
    # 设置训练器
    trainer = setup_trainer(model, train_dataset, eval_dataset)
    
    # 启动训练
    print("开始模型微调...")
    trainer.train()
    print("模型微调完成！")
    
    # 评估模型
    test_results = evaluate_model(trainer, test_dataset)
    
    # 保存模型
    save_model(model, tokenizer)
    
    # 推理示例
    sample_texts = [
        "这家酒店的服务太棒了，房间也很干净，下次还来！",
        "位置很难找，设施也很陈旧，不会再住了。",
        "中规中矩吧，没什么特别的优点，但也没什么缺点。",
        "床不是很舒服，但是早餐非常丰盛，总体还行。"
    ]
    inference_example(OUTPUT_DIR, OUTPUT_DIR, sample_texts)


if __name__ == '__main__':
    main()