#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;

const int N = 1e5 + 10;

ll n;
char s[N][N];
ll pre[N];

void init(str t) // next数组求值
{
    ll lent = t.size(); // 子串
    ll j = 0;
    for (int i = 2; i <= lent; i++)
    { // 判断当前位置是否能够发生匹配，跳转到可以匹配的位置
        while (j && t[i] != t[j + 1])
            j = pre[j]; // 前面有位置匹配上，后一位匹配不上，向前跳转一位
        // 如果当前两个位置字母相同
        if (t[i] == t[j + 1])
            j++;
        // 储存当前位置匹配失败所跳转到的地方
        pre[i] = j;
    }
}

/*
bool KMP(char t[N])
{
    ll lens = strlen(s + 1); // 主串
    ll lent = strlen(t + 1); // 字串
    for (int i = 1, j = 1; i <= lens; i++)
    {
        // 字符匹配失败，根据next数组跳过部分字符
        while (j && s[i] != t[j + 1])
            j = pre[j];
        if (s[i] == t[j + 1]) // 字符匹配，指针后移
            j++;
        if (j == lent) // 匹配成功
            return 1;
    }
    return 0;
}
*/

int main()
{
    cin >> n;
    for (int i = 1; i < n; i++)
    {
        cin >> s[i];
        str nows=s[i];
        nows += s[i + 1];
        init(nows);
    }
    return 0;
}