#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef double dou;
typedef string str;
typedef pair<ll, ll> pll;
#define pb push_back
// pl/pr 用于快速获取子节点编号（基于完全二叉树的数组表示）
// pl(p) -> 左子节点 = p << 1
// pr(p) -> 右子节点 = p << 1 | 1
#define pl(x) (x << 1)
#define pr(x) (x << 1 | 1)

// N 为原始数组最大可能大小的上界（线段树数组大小会基于此分配）
// 注意：本实现假定输入数组下标为 1..n（即使用 1-based 下标管理区间）
const int N = 2e5 + 10;

ll t;           // 用于多组测试的计数（solve 中会使用）
ll tree[4 * N]; // 线段树数组，存储区间和（或节点信息）
ll lazy[4 * N]; // 懒标记数组，存储对节点尚未下传的“加值”

// pushdown：将节点 p 的懒标记下传到左右子节点，
// 参数 p: 当前节点编号，l,r: 当前节点表示的区间 [l,r]
// 该函数会更新左右子节点的 tree 和 lazy，使得当前节点的懒标记清零
void pushdown(ll p, ll l, ll r)
{
    // mid 是当前区间的中点，用于计算左右子区间长度
    ll mid = (l + r) >> 1;
    // 将当前懒标记的加值作用到左子区间和右子区间的节点和上
    tree[pl(p)] += lazy[p] * (mid - l + 1); // 更新左子节点的区间和
    tree[pr(p)] += lazy[p] * (r - mid);     // 更新右子节点的区间和
    // 将懒标记累加到左右子节点的 lazy 中（下次访问子节点时会继续传下去）
    lazy[pl(p)] += lazy[p]; // 传递给左子节点
    lazy[pr(p)] += lazy[p]; // 传递给右子节点
    // 清空当前节点的懒标记（已下传）
    lazy[p] = 0;
}

// update：在区间 [ul,ur] 上加上 val
// 参数说明：p: 当前节点编号，ul/ur: 更新区间，val: 加值，l/r: 当前节点表示的区间
// 重要点：函数既处理完全覆盖的情况（直接累加懒标记并返回），
// 也处理部分覆盖：先下传懒标记，更新当前节点对应的交集部分，然后递归更新子节点。
// 注意：本实现以当前节点的交集长度直接更新 tree[p]，然后递归再修正子树。
// 这种做法可以在不返回子树求和结果的情况下保持父节点和的一致性。
void update(ll p, ll ul, ll ur, ll val, ll l, ll r)
{
    if (ul > r || ur < l) // 无交集，直接返回
        return;
    if (ul <= l && ur >= r) // 当前节点区间完全被更新区间覆盖
    {
        // 直接增加当前节点的区间和，并累加懒标记（延迟下传）
        tree[p] += val * (r - l + 1); // 节点和按区间长度增加
        lazy[p] += val;               // 设置懒标记
        return;
    }
    // 若当前节点有懒标记且非叶节点，需要先下传到子节点，保证子节点数据正确
    if (lazy[p]) // 有懒标记需要下传
        pushdown(p, l, r);

    // 计算当前节点区间与更新区间的交集 [mxl, mxr]
    // 这一步用于先行更新 tree[p]（父节点的和），再递归更新子节点
    ll mxl = max(ul, l);
    ll mxr = min(ur, r);
    // 如果有交集，则按交集长度更新当前节点的和（注意交集可能为空，但上面已排除无交集情况）
    tree[p] += val * (mxr - mxl + 1); // 更新当前节点（仅交集部分）
    if (l == r)                       // 若是叶节点，已完成更新
        return;
    // 递归更新左右子区间
    ll mid = (l + r) >> 1;
    update(pl(p), ul, ur, val, l, mid);     // 更新左子树
    update(pr(p), ul, ur, val, mid + 1, r); // 更新右子树
    // 注：此处不需要再合并子节点到父节点，因为父节点已按交集长度直接更新，
    // 但若改为常规实现（不在父节点先更新交集），此处应设置 tree[p] = tree[pl(p)] + tree[pr(p)]
}

// query：查询区间 [ql,qr] 的和
// 参数说明：p: 当前节点编号，ql/qr: 查询区间，l/r: 当前节点表示的区间
// 递归查询，遇到懒标记先下传，遇到完全覆盖则直接返回 tree[p]。
ll query(ll p, ll ql, ll qr, ll l, ll r)
{
    if (ql > r || qr < l) // 无交集，返回中性元素（和为 0）
        return 0;
    if (ql <= l && qr >= r) // 当前节点区间被查询区间完全包含
        return tree[p];
    // 若当前节点存在懒标记，先下传，保证子节点数据正确
    if (lazy[p]) // 有懒标记需要下传
        pushdown(p, l, r);
    ll mid = (l + r) >> 1;
    ll res = 0;
    // 递归查询左右子区间（可能部分覆盖）
    res += query(pl(p), ql, qr, l, mid);     // 查询左子树
    res += query(pr(p), ql, qr, mid + 1, r); // 查询右子树
    return res;
}

// solve：示例解题流程，读取输入并使用上面的 update/query 接口
// 注意：输入假定为 1-based 下标（即数组元素从 1 到 n），与实现保持一致
void solve()
{
    ll n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        ll x;
        cin >> x;
        // 初始化时，将单点区间 [i,i] 加上 x，填充线段树
        update(1, i, i, x, 1, n); // 初始化线段树（1-based）
    }
    while (m--)
    {
        ll op;
        cin >> op;
        if (op == 1)
        {
            ll l, r, val;
            cin >> l >> r >> val;
            // 区间加值（输入应为 1..n 的区间）
            update(1, l, r, val, 1, n);
        }
        else if (op == 2)
        {
            ll l, r;
            cin >> l >> r;
            // 区间查询并输出结果
            cout << query(1, l, r, 1, n) << endl;
        }
    }
}

int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    t = 1;
    // cin >> t;
    while (t--)
    {
        solve();
    }
    return 0;
}