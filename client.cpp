#include <iostream>
#include <string>
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")

int main() {
    SetConsoleOutputCP(65001);  // 设置输出为 UTF-8
    SetConsoleCP(65001);        // 设置输入为 UTF-8

    std::string server_ip = "113.44.106.189";
    int server_port = 12345;

    // 初始化 Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSA 初始化失败" << std::endl;
        return 1;
    }

    while (true) {
        std::string message;
        std::cout << "请输入要发送的评论 (输入 'quit' 退出): ";
        std::getline(std::cin, message);

        if (message == "quit" || message == "exit" || message == "q") {
            std::cout << "退出客户端." << std::endl;
            break;
        }

        // 创建 socket
        SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == INVALID_SOCKET) {
            std::cerr << "创建 socket 失败" << std::endl;
            continue;
        }

        // 服务器地址
        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(server_port);
        serverAddr.sin_addr.s_addr = inet_addr(server_ip.c_str());

        // 连接服务器
        if (connect(sock, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            std::cerr << "连接服务器失败: " << WSAGetLastError() << std::endl;
            closesocket(sock);
            continue;
        }

        // 发送消息
        send(sock, message.c_str(), message.size(), 0);

        // 接收服务器回复
        char buffer[1024] = {0};
        int bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            std::cout << "接收回复: " << buffer << std::endl;
        } else {
            std::cerr << "接收失败或服务器断开连接" << std::endl;
        }

        // 关闭连接
        closesocket(sock);
    }

    WSACleanup();
    return 0;
}
