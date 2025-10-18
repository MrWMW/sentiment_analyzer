import socket
from sentiment_ana import SentimentAnalyzer
import argparse
import logging
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 创建自定义日志过滤器添加客户端IP
class ClientIPFilter(logging.Filter):
    def filter(self, record):
        record.client_IP = getattr(record, 'client_IP', 'N/A')
        return True

# 配置基础日志设置
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('translate_test.log', mode='a')
formatter = logging.Formatter(
    '%(client_IP)s-%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
file_handler.addFilter(ClientIPFilter())
logger.addHandler(file_handler)

#控制台输出
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.addFilter(ClientIPFilter())
logger.addHandler(console_handler)

def recv_data(sock,client_ip):
    data = sock.recv(1024).decode('utf-8')
    logging.info(f"接收到数据: {data}",extra={'client_IP': client_ip})
    return data

def send_data(sock, message, client_ip):
    if(message["sentiment"] == "好评 (Positive)"):
        message = "好评 (Positive)"
    else:
        message = "差评 (Negative)"
    sock.send(message.encode('utf-8'))
    logging.info(f"发送数据: {message}",extra={'client_IP': client_ip})

def main():
    parser = argparse.ArgumentParser(description="情感分析服务器")
    parser.add_argument('--model_path', type=str, default='./output/bert-chnsenticorp-finetuned', help='预训练模型名称')
    parser.add_argument('--port', type=int, default=12345, help='服务器端口号')
    args = parser.parse_args()

    logging.info("="*50,extra={'client_IP': 'SERVER'})
    logging.info(f"加载模型：{args.model_path}",extra={'client_IP': 'SERVER'})

    sentiment_analyzer = SentimentAnalyzer(args.model_path, max_length=1024)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    server_socket.bind((host, args.port))
    server_socket.listen(10)
    logging.info(f"服务器启动，ip地址{host}，监听端口 {args.port}...")

    while True:
        try:
            client_socket,client_address = server_socket.accept()
            client_ip = client_address[0]
            logging.info(f"连接来自 {client_address}",extra={'client_IP': client_ip})
            data = recv_data(client_socket, client_ip)
            logging.info(f"开始情感分析...",extra={'client_IP': client_ip})
            sentiment = sentiment_analyzer.predict_single(data)
            logging.info(f"情感分析结果: {sentiment}",extra={'client_IP': client_ip})
            send_data(client_socket,sentiment, client_ip)
            client_socket.close()
        
        except Exception as e:
            logging.error(f"发生错误: {e}",extra={'client_IP': client_ip})
            if 'client_socket' in locals():
                client_socket.close()

if __name__ == "__main__":
    main()