#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

const int N = 1e8;

vector<int> primes;
bool is_prime[N + 10];

void linearSieve(int n)
{
    memset(is_prime, 1, sizeof(is_prime));
    is_prime[0] = is_prime[1] = false;
    primes.clear();

    for (int i = 2; i <= n; i++)
    {
        if (is_prime[i])
            primes.push_back(i);
        for (int j = 0; j < (int)primes.size() && i <= n / primes[j]; j++)
        {
            is_prime[i * primes[j]] = false;
            if (i % primes[j] == 0)
                break;
        }
    }
}

int main()
{
    linearSieve(N);

    int t, n;
    cin >> t;
    while (t--)
    {
        cin >> n;
        cout << (is_prime[n] ? "Yes" : "No") << endl;
    }
}
