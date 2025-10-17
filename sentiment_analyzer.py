# -*- coding: utf-8 -*-
"""
made by Mr.W
话题情感倾向分析模型
用于分析一个话题下所有评论的情感倾向，并基于话题整体情感倾向进行个体评论分析
根据话题整体倾向调整个体评论的情感倾向
"""

import os
import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


class TopicSentimentAnalyzer:
    """话题情感分析器"""
    
    def __init__(self, model_path, max_length=128, batch_size=16):
        """
        初始化话题情感分析器
        
        参数:
            model_path: 模型路径
            max_length: 最大序列长度
            batch_size: 批量处理大小
        """
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """加载模型和分词器"""
        print(f"正在从 '{self.model_path}' 加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        print(f"模型已加载到: {self.device}")
        
        # 标签映射
        self.id2label = {0: "消极", 1: "积极"}
    
    def predict_sentiment_probs(self, comments):
        """
        预测评论的情感概率分布
        
        参数:
            comments: 评论列表
            
        返回:
            情感概率矩阵 (n_comments, n_classes)
        """
        all_probs = []
        
        # 分批处理评论
        for i in tqdm(range(0, len(comments), self.batch_size), desc="计算情感概率"):
            batch_comments = comments[i:i+self.batch_size]
            
            # 过滤空评论
            valid_comments = [comment for comment in batch_comments if isinstance(comment, str) and comment.strip()]
            if not valid_comments:
                continue
            
            # 分词和编码
            inputs = self.tokenizer(
                valid_comments,
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
            
            # 获取概率分布
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
        
        return np.array(all_probs)
    
    def calculate_topic_sentiment_baseline(self, probs):
        """
        计算话题情感倾向基准
        
        参数:
            probs: 所有评论的情感概率矩阵
            
        返回:
            话题情感基准 (积极概率基准值)
        """
        # 计算每条评论的积极概率
        positive_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        
        # 计算话题整体情感倾向（加权平均）
        topic_baseline = np.mean(positive_probs)
        
        print(f"话题情感基准值: {topic_baseline:.4f}")
        if topic_baseline > 0.6:
            print("话题整体情感倾向: 积极")
        elif topic_baseline < 0.4:
            print("话题整体情感倾向: 消极")
        else:
            print("话题整体情感倾向: 中性")
            
        return topic_baseline
    
    def adjust_sentiment_based_on_topic(self, probs, topic_baseline):
        """
        基于话题整体倾向调整情感倾向
        
        参数:
            probs: 情感概率矩阵
            topic_baseline: 话题情感基准
            
        返回:
            调整后的情感分类列表
        """
        adjusted_sentiments = []
        adjustment_factors = []
        
        # 根据话题整体倾向调整情感判断
        for i, prob in enumerate(probs):
            positive_prob = prob[1] if len(prob) > 1 else prob[0]
            
            # 根据话题整体倾向进行调整
            if topic_baseline > 0.6:  # 话题整体积极
                # 对消极评论进行增加以让其倾向为积极
                if positive_prob < 0.4:  # 原始判断为消极
                    adjustment = 0.3 * (1 - positive_prob)  # 调整因子，消极程度越高调整越大
                    adjusted_prob = positive_prob + adjustment
                    adjustment_factors.append(adjustment)
                else:
                    adjusted_prob = positive_prob
                    adjustment_factors.append(0)
            
            elif topic_baseline < 0.4:  # 话题整体消极
                # 对积极评论进行减少以让其倾向为消极
                if positive_prob > 0.6:  # 原始判断为积极
                    adjustment = 0.3 * positive_prob  # 调整因子，积极程度越高调整越大
                    adjusted_prob = positive_prob - adjustment
                    adjustment_factors.append(adjustment)
                else:
                    adjusted_prob = positive_prob
                    adjustment_factors.append(0)
            
            else:  # 话题整体中性
                adjusted_prob = positive_prob
                adjustment_factors.append(0)
            
            # 根据调整后的概率确定情感分类
            if adjusted_prob > 0.6:
                adjusted_sentiment = "积极"
            elif adjusted_prob < 0.4:
                adjusted_sentiment = "消极"
            else:
                adjusted_sentiment = "中性"
                
            adjusted_sentiments.append(adjusted_sentiment)
        
        return adjusted_sentiments, adjustment_factors
    
    def process_csv(self, input_file, text_column="评论", output_dir="./result"):
        """
        处理CSV文件中的评论
        
        参数:
            input_file: 输入CSV文件路径
            text_column: 包含评论文本的列名
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        try:
            df = pd.read_csv(input_file)
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return
        
        # 检查指定的文本列是否存在
        if text_column not in df.columns:
            print(f"错误: CSV文件中没有名为 '{text_column}' 的列")
            print(f"可用列: {list(df.columns)}")
            return
        
        # 提取评论
        comments = df[text_column].tolist()
        print(f"找到 {len(comments)} 条评论")
        
        # 分析评论情感概率
        probs = self.predict_sentiment_probs(comments)
        
        if len(probs) == 0:
            print("没有分析出任何结果")
            return
        
        # 计算话题情感基准
        topic_baseline = self.calculate_topic_sentiment_baseline(probs)
        
        # 基于话题基准调整情感倾向
        adjusted_sentiments, adjustment_factors = self.adjust_sentiment_based_on_topic(probs, topic_baseline)
        
        # 获取绝对情感分类（原始模型预测）
        absolute_sentiments = []
        absolute_confidences = []
        for prob in probs:
            if len(prob) > 1:
                pred = np.argmax(prob)
                confidence = prob[pred]
                absolute_sentiments.append(self.id2label[pred])
            else:
                # 二分类特殊情况处理
                pred = 1 if prob[0] > 0.5 else 0
                confidence = prob[0] if pred == 1 else 1 - prob[0]
                absolute_sentiments.append(self.id2label[pred])
            absolute_confidences.append(confidence)
        
        # 计算调整后的积极概率
        adjusted_positive_probs = []
        for i, prob in enumerate(probs):
            positive_prob = prob[1] if len(prob) > 1 else prob[0]
            
            if topic_baseline > 0.6 and positive_prob < 0.5:
                adjusted_positive_probs.append(positive_prob + adjustment_factors[i])
            elif topic_baseline < 0.4 and positive_prob > 0.5:
                adjusted_positive_probs.append(positive_prob - adjustment_factors[i])
            else:
                adjusted_positive_probs.append(positive_prob)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            "评论内容": comments[:len(probs)],
            "绝对情感分类": absolute_sentiments,
            "绝对情感置信度": absolute_confidences,
            "调整后情感分类": adjusted_sentiments,
            "原始积极概率": probs[:, 1] if probs.shape[1] > 1 else probs[:, 0],
            "调整后积极概率": adjusted_positive_probs,
            "调整因子": adjustment_factors,
            "话题情感基准": [topic_baseline] * len(probs)
        })
        
        # 合并原始数据和情感分析结果
        final_df = df.copy().iloc[:len(results_df)]
        final_df = pd.concat([final_df, results_df[["绝对情感分类", "绝对情感置信度", "调整后情感分类", 
                                                   "原始积极概率", "调整后积极概率", "调整因子", "话题情感基准"]]], axis=1)
        
        # 分离不同情感类型的评论
        positive_df = final_df[final_df["调整后情感分类"] == "积极"]
        negative_df = final_df[final_df["调整后情感分类"] == "消极"]
        neutral_df = final_df[final_df["调整后情感分类"] == "中性"]
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        positive_file = os.path.join(output_dir, f"{base_name}_调整后积极评论.csv")
        negative_file = os.path.join(output_dir, f"{base_name}_调整后消极评论.csv")
        neutral_file = os.path.join(output_dir, f"{base_name}_调整后中性评论.csv")
        all_file = os.path.join(output_dir, f"{base_name}_全部评论_processed2.csv")
        
        # 保存结果
        positive_df.to_csv(positive_file, index=False, encoding='utf-8-sig')
        negative_df.to_csv(negative_file, index=False, encoding='utf-8-sig')
        neutral_df.to_csv(neutral_file, index=False, encoding='utf-8-sig')
        final_df.to_csv(all_file, index=False, encoding='utf-8-sig')
        
        print(f"分析完成!")
        print(f"调整后积极评论: {len(positive_df)} 条, 已保存到: {positive_file}")
        print(f"调整后消极评论: {len(negative_df)} 条, 已保存到: {negative_file}")
        print(f"调整后中性评论: {len(neutral_df)} 条, 已保存到: {neutral_file}")
        print(f"全部评论: {len(final_df)} 条, 已保存到: {all_file}")
        
        # 显示话题情感统计
        print(f"\n话题情感分析统计:")
        print(f"话题情感基准值: {topic_baseline:.4f}")
        if topic_baseline > 0.6:
            print("话题整体情感倾向: 积极")
        elif topic_baseline < 0.4:
            print("话题整体情感倾向: 消极")
        else:
            print("话题整体情感倾向: 中性")
        
        # 显示调整前后情感分布对比
        print(f"\n情感分布对比:")
        original_positive = sum(1 for s in absolute_sentiments if s == "积极")
        original_negative = sum(1 for s in absolute_sentiments if s == "消极")
        adjusted_positive = len(positive_df)
        adjusted_negative = len(negative_df)
        
        print(f"原始积极评论: {original_positive} ({original_positive/len(final_df):.2%})")
        print(f"原始消极评论: {original_negative} ({original_negative/len(final_df):.2%})")
        print(f"调整后积极评论: {adjusted_positive} ({adjusted_positive/len(final_df):.2%})")
        print(f"调整后消极评论: {adjusted_negative} ({adjusted_negative/len(final_df):.2%})")
        
        # 显示调整效果
        if topic_baseline > 0.6:
            print(f"\n调整效果: 将 {adjusted_positive - original_positive} 条评论从消极调整为积极")
        elif topic_baseline < 0.4:
            print(f"\n调整效果: 将 {original_positive - adjusted_positive} 条评论从积极调整为消极")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="话题情感分析批处理脚本")
    parser.add_argument("--input", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--model_path", type=str, default="./output/bert-chnsenticorp-finetuned",
                       help="微调模型路径")
    parser.add_argument("--text_column", type=str, default="评论",
                       help="CSV文件中包含评论文本的列名")
    parser.add_argument("--output_dir", type=str, default="./result",
                       help="输出目录")
    parser.add_argument("--max_length", type=int, default=128,
                       help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批量处理大小")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = TopicSentimentAnalyzer(
        model_path=args.model_path,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    
    # 处理CSV文件
    analyzer.process_csv(
        input_file=args.input,
        text_column=args.text_column,
        output_dir=args.output_dir
    )


if __name__ == "__main__":

    main()
