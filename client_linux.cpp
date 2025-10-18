#include <iostream>
#include <string>
#include <cstring>      // for memset
#include <unistd.h>     // for close()
#include <arpa/inet.h>  // for socket(), connect(), inet_addr()
#include <netinet/in.h> // for sockaddr_in

int main() {
    std::string server_ip = "113.44.106.189";  // 你的服务器 IP
    int server_port = 12345;

    while (true) {
        std::string message;
        std::cout << "请输入要发送的评论 (输入 'quit' 退出): ";
        std::getline(std::cin, message);

        if (message == "quit" || message == "exit" || message == "q") {
            std::cout << "退出客户端." << std::endl;
            break;
        }

        // 创建 socket
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "创建 socket 失败" << std::endl;
            continue;
        }

        // 配置服务器地址
        sockaddr_in serverAddr;
        memset(&serverAddr, 0, sizeof(serverAddr));
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(server_port);
        serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());

        // 连接服务器
        if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "连接服务器失败" << std::endl;
            close(sock);
            continue;
        }

        // 发送数据
        ssize_t sent = send(sock, message.c_str(), message.size(), 0);
        if (sent < 0) {
            std::cerr << "发送失败" << std::endl;
            close(sock);
            continue;
        }

        // 接收数据
        char buffer[1024];
        memset(buffer, 0, sizeof(buffer));
        ssize_t received = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (received > 0) {
            buffer[received] = '\0';
            std::cout << "接收回复: " << buffer << std::endl;
        } else {
            std::cerr << "接收失败或服务器断开连接" << std::endl;
        }

        close(sock);
    }

    return 0;
}
