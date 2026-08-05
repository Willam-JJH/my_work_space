#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <utility>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;

const int N = 5e3 + 10;

int n, a, b, leng;
bool vis[N];
int sum[N];
vector<int> Primes;
vector<int> res;

void sieve(int x)
{
    memset(vis, 1, sizeof(vis));
    for (int i = 2; i <= x; i++)
    {
        if (vis[i])
        {
            Primes.push_back(i);
            for (int j = i * 2; j <= x; j += i)
                vis[j] = 0;
        }
    }
    leng = Primes.size();
}

int cul1(int x, int p)
{
    int cnt = 0;
    while (x)
    {
        cnt += x / p;
        x /= p;
    }
    return cnt;
}

void cul2()
{
    for (int i = 0; i < leng; i++)
    {
        for (int j = 0; j < sum[i]; j++)
        {
            int t = 0;
            for (int k = 0; k < res.size(); k++)
            {
                t += res[k] * Primes[i];
                res[k] = t % 10;
                t /= 10;
            }
            while (t)
            {
                res.push_back(t % 10);
                t /= 10;
            }
        }
    }
}

int main()
{
    cin >> a >> b;
    sieve(a);
    for (int i = 0; i < leng; i++)
    {
        int p = Primes[i];
        sum[i] = cul1(a, p) - cul1(b, p) - cul1(a - b, p);
    }
    res.push_back(1);
    cul2();
    for (int i = res.size() - 1; i >= 0; i--)
        cout << res[i];
    return 0;
}