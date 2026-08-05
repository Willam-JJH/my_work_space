#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;

const int N = 1e2 + 10;

ll n;
dou a[N][N];

int f()
{
    ll r = 1;
    for (int c = 1; c <= n; c++)
    {
        ll pos = r;
        for (int i = r; i <= n; i++)
            if (abs(a[i][c]) > abs(a[pos][c]))
                pos = i;
        if (abs(a[pos][c]) < 1e-8)
            continue;
        for (int i = 1; i <= n + 1; i++)
            swap(a[r][i], a[pos][i]);
        for (int i = n + 1; i >= c; i--)
            a[r][i] /= a[r][c];
        for (int i = r + 1; i <= n; i++)
        {
            if (abs(a[i][c]) <= 1e-8)
                continue;
            for (int j = n + 1; j >= c; j--)
            {
                a[i][j] -= a[r][j] * a[i][c];
            }
        }
        r++;
    }
    if (r <= n)
    {
        for (int i = r; i <= n; i++)
        {
            if (abs(a[i][n + 1]) > 1e-8)
                return 2;
        }
        return 1;
    }
    for (int i = n; i > 0; i--)
    {
        for (int j = i + 1; j <= n; j++)
            a[i][n + 1] -= a[i][j] * a[j][n + 1];
    }
    return 0;
}

void solve()
{
    cin >> n;
    for (int i = 1; i <= n + 1; i++)
        for (int j = 1; j <= n; j++)
            cin >> a[i][j];
    int t = f();
    if (t == 0)
    {
        for (int i = 1; i <= n; i++)
        {
            if (abs(a[i][n + 1]) < 1e-8)
                a[i][n + 1] = 0.0;
            printf("%.2lf\n", a[i][n + 1]);
        }
    }
    else if (t == 1)
        cout << "Infinite group solutions";
    else
        cout << "No solution";
}

int main()
{
    /*
    freopen(".in", "r", stdin);
    freopen(".out", "w", stdout);
    */
    solve();
    /*
    fclose(stdin);
    fclose(stdout);
    */
    return 0;
}