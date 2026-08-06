#include "C++heads.h"
// #include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;
#define lowbit(x) (x & -x)

const int N = 50;

bool flag = 1; // 判断是否退出程序的变量
int mode;      // 每次的操作
// 学生的名字列表
std::string namelist[N] = {
    /*
    用于储存同学们的姓名
    如"张三","李四", "王五" 等
    这里为了保护同学隐私将名字删除，不影响程序的编译、运行
    */
};

struct node
{
    int score;
    std::string name;
} v[N]; // 储存同学的分数、名字

void f1() // 分数查询功能
{
    std::cout << "\n你需要查询全班学生的分数情况或单个同学的分数情况?\n";
    std::cout << "1.全班同学 2.单个同学\n\n";
    int mode1;
    std::cin >> mode1; // 选择全班同学还是某一个同学
    if (mode1 == 1)
    { // 全班同学的情况
        std::cout << "\n以下为全班分数情况:\n";
        for (int i = 1; i < 49; i++)
        {
            if (i < 10)
                std::cout << 0;
            std::cout << i << "号：" << v[i].name << v[i].score << " 分\n";
        }
    }
    else if (mode1 == 2)
    {
        std::cout << "\n请输入要查询的学生的学号:\n\n";
        int numb;
        std::cin >> numb;
        if (numb > N - 1 || numb < 1)
        { // 防止坏孩子乱输指令
            std::cout << "\n班上无此同学。\n\n";
            return;
        }
        std::cout << "\n"
                  << numb << "号：" << v[numb].score << " 分\n";
    }
    else
    { // 还是防止坏孩子乱输入指令。
        std::cout << "\n\n没有此选项,请重新输入。\n\n";
        return;
    }
}

void f2() // 加减分数功能
{
    std::cout << "\n请输入要加分或减分的学生的学号。\n\n";
    int numb;
    std::cin >> numb;
    if (numb > N - 1 || numb < 1)
    { // 还是防止坏孩子乱输入。
        std::cout << "\n班上无此同学,请重新输入。\n\n";
        return;
    }
    std::cout << "\n请输入" << numb << "号要加的分数，如需减分请输入负数。\n\n";
    int plus;
    std::cin >> plus;
    v[numb].score += plus;
    std::cout << "\n操作完成,当前 " << numb << " 号分数为：" << v[numb].score << "分\n";
}

void f3() // 存档码读入功能
{
    std::cout << "\n请输入存档码:\n";
    std::cout << "(如果要退出功能请输入0)\n\n";
    std::string code;
    std::cin >> code;
    if (code == "0")
    {
        std::cout << "已退出。\n";
        return;
    }
    else if (code.size() != 4 * N)
    {
        std::cout << "存档码长度不合法。\n";
        return;
    }
    for (int i = 0; i < N; i++)
    {
        char zhengfu = code[i * 4];
        int bai = code[i * 4 + 1] - '0'; // v[i]的百位
        int shi = code[i * 4 + 2] - '0'; // v[i]的十位
        int ge = code[i * 4 + 3] - '0';  // v[i]的个位
        if (zhengfu == '-')
            v[i + 1].score = -(bai * 100 + shi * 10 + ge);
        else
            v[i + 1].score = bai * 100 + shi * 10 + ge;
    }
    std::cout << "\n存档码读取完毕。\n";
}

void f4() // 存档码输出功能
{
    // 这里其实就是把数组v的每一项输出，美其名曰存档码，如果加个转16进制可能更像。
    std::cout << "\n存档码如下:\n\n";
    for (int i = 1; i < N; i++)
    {
        if (v[i].score < 0)
            std::cout << '-';
        else
            std::cout << '+';
        // 输出前导零
        if (abs(v[i].score) < 100 && abs(v[i].score) >= 10)
            std::cout << '0';
        else if (abs(v[i].score) < 10)
            std::cout << "00";
        std::cout << abs(v[i].score);
    }
    std::cout << "\n\n存档码已输出完成。\n";
    flag = 0;
}

int main()
{
    // 初始化每个同学的姓名
    for (int i = 1; i < N; i++)
    {
        v[i].name = namelist[i - 1];
        v[i].score = 0;
    }
    // 开始进行分数查询
    do
    {
        std::cout << "你需要进行以下的哪一项操作？\n";
        std::cout << "1.分数查询 2.分数加减\n";
        std::cout << "3.读入存档码 4.获取存档码并退出程序\n\n";
        std::cin >> mode; // 储存每一次的操作选项
        if (mode == 1)
            f1();
        else if (mode == 2)
            f2();
        else if (mode == 3)
            f3();
        else if (mode == 4)
            f4();
        else
        { // 还是防坏孩子乱输指令。
            std::cout << "\n无此操作,请重新输入。\n\n";
            continue;
        }
        if (flag)
            std::cout << "\n\n";
    } while (flag);
    std::cout << "已退出程序。";
    return 0;
}