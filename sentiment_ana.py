# -*- coding: utf-8 -*-
"""
微调模型推理脚本
用于加载和使用微调后的中文情感分析模型进行推理
"""

import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# 配置基础日志设置
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
    filename='sentiment_ana.log',  # 输出到文件（可选）
    filemode='a'  # 文件模式：'a'追加，'w'覆盖
)

class SentimentAnalyzer:
    """情感分析器类，封装模型加载和推理功能"""
    
    def __init__(self, model_path, max_length=128):
        """
        初始化情感分析器
        
        参数:
            model_path: 模型路径
            max_length: 最大序列长度
        """
        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        print(f"正在从 '{self.model_path}' 加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        print("模型加载完成!")
        
        # 标签映射
        self.id2label = {0: "差评 (Negative)", 1: "好评 (Positive)"}
    
    def predict(self, texts, batch_size=8):
        """
        对文本进行情感分析
        
        参数:
            texts: 文本列表或单个文本
            batch_size: 批量处理大小
            
        返回:
            预测结果列表
        """
        # 确保输入是列表形式
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 分词和编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取预测结果
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # 转换为标签
            for text, pred in zip(batch_texts, predictions):
                sentiment = self.id2label[pred]
                confidence = torch.softmax(logits, dim=-1)[0][pred].item()
                results.append({
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 4),
                    "label": pred
                })
        
        return results
    
    def predict_single(self, text):
        """
        对单个文本进行情感分析
        
        参数:
            text: 输入文本
            
        返回:
            预测结果字典
        """
        return self.predict(text)[0]
    
    def print_results(self, results):
        """打印预测结果"""
        for result in results:
            print(f"评论: {result['text']}")
            print(f"  情感: {result['sentiment']}")
            print(f"  置信度: {result['confidence']:.2%}")
            print("-" * 50)
        for result in results:
            logging.info(f"评论: {result['text']} | 情感: {result['sentiment']} | 置信度: {result['confidence']:.2%}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文情感分析推理脚本")
    parser.add_argument("--model_path", type=str, default="./output/bert-chnsenticorp-finetuned",
                       help="微调模型路径")
    parser.add_argument("--text", type=str, help="要分析的文本")
    parser.add_argument("--file", type=str, help="包含文本的文件路径")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批量处理大小")
    
    args = parser.parse_args()
    
    logging.info("启动中文情感分析器")


    # 创建情感分析器
    analyzer = SentimentAnalyzer(args.model_path, args.max_length)
    
    # 处理输入
    if args.text:
        # 分析单个文本
        result = analyzer.predict_single(args.text)
        analyzer.print_results([result])
    
    elif args.file:
        # 从文件读取文本
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            if not texts:
                print("文件为空或没有有效文本")
                return
            
            print(f"从文件 '{args.file}' 读取了 {len(texts)} 条评论")
            results = analyzer.predict(texts, args.batch_size)
            analyzer.print_results(results)
            
        except FileNotFoundError:
            print(f"错误: 文件 '{args.file}' 不存在")
            logging.error(f"错误: 文件 '{args.file}' 不存在")
        except Exception as e:
            print(f"读取文件时出错: {e}")
            logging.error(f"读取文件时出错：{e}")
    
    else:
        try:
        # 交互模式
            print("中文情感分析器已启动 (输入 'quit' 或 'exit' 退出)")
            print("-" * 50)
        
            while True:
                text = input("请输入要分析的评论: ").strip()
                logging.info(f"用户输入: {text}")
                if text.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    logging.info(f"关闭模型")
                    break
            
                if not text:
                    continue
            
                result = analyzer.predict_single(text)
                analyzer.print_results([result])
                print()
        except Exception as e:
            print(f"读取评论时出现错误：{e}")
            logging.error(f"读取评论时出现错误：{e}")


if __name__ == "__main__":
    main()