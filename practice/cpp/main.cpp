#include "C++heads.h"
// #include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;
#define lowbit(x) (x & -x)

const int N = 1e2 + 10;

ll a[N];
ll p[N];

void init()
{
}

void solve()
{
    ll n;
    scanf("%lld", &n);
    for (int i = 1; i <= n; i++)
        scanf("%lld", &a[i]);
    partial_sum(a, a + n, p);
    for (int i = 1; i <= n; i++)
        printf("%lld", p[i]);
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