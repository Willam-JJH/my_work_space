#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;

const int N = 1e3 + 9;

ll ans;
ll n, m;
ll u, v, w;

ll dis[N];
bool vis[N];

struct node{
    ll v, w;
};

vector<node> mp[2][N];

void dij(ll s)
{
    memset(vis, 0, sizeof(vis));
    memset(dis, 0x3f, sizeof(dis));
    dis[1] = 0;
    for (int i = 1; i < n; i++)
    {
        ll u = -1;
        for (int j = 1; j <= n; j++)
        {
            if(!vis[j]){
                if (u == -1 || dis[j] < dis[u])
                    u = j;
            }
        }
        if (u == -1)
            break;
        vis[u] = 1;
        for (int j = 0; j < mp[s][u].size(); j++)
        {
            ll y = mp[s][u][j].v;
            ll w = mp[s][u][j].w;
            if (dis[y] > dis[u] + w)
                dis[y] = dis[u] + w;
        }
    }
}

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= m; i++)
    {
        cin >> u >> v >> w;
        mp[0][u].push_back({v, w});
        mp[1][v].push_back({u, w});
    }
    dij(0);
    for (int i = 1; i <= n; i++)
        ans += dis[i];
    dij(1);
    for (int i = 1; i <= n; i++)
        ans += dis[i];
    cout << ans;
    return 0;
}