#include "C++heads.h"
// #include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;
#define lowbit(x) (x & -x)

const int N = 1e3 + 10;

ll a[N];
ll pre[N];

void init()
{
}

void solve()
{
    memset(a, 0, sizeof(a));
    memset(pre, 0, sizeof(pre));
    ll n, ans = 0;
    scanf("%lld", &n);
    for (int i = 1; i <= n; i++)
    {
        char ch;
        scanf(" %c", &ch);
        a[i] = ch - '0';
    }
    partial_sum(a + 1, a + n + 1, pre + 1);
    for (int i = 1; i <= n; i++)
    {
        for (int j = i; j <= n; j++)
        {
            if (pre[j] - pre[i - 1] == j - i + 1)
                ans++;
        }
    }
    printf("%lld\n", ans);
}

int main()
{
    // freopen(".in", "r", stdin);
    // freopen(".out", "w", stdout);
    ll t = 1;
    scanf("%lld", &t);
    init();
    while (t--)
        solve();
    return 0;
}