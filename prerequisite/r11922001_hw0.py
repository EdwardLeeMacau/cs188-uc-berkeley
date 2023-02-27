from typing import List, Set

class Solution:
    islands: List[Set[int]]
    N: int
    M: int

    def __init__(self):
        # input() is line terminated
        problem_size = input().strip()

        # island uses 1-based indexing
        self.N, self.M = map(int, problem_size.split(' '))
        self.islands = [ set() for _ in range(self.N + 1) ]

        # Undirected graph
        for _ in range(self.M):
            bridge = input().strip()
            src, dest = map(int, bridge.split(' '))
            self.islands[src].add(dest)
            self.islands[dest].add(src)

    def __str__(self):
        ans = ""

        for idx in range(1, self.N + 1):
            ans += f'{idx:2d}: {self.islands[idx]}\n'

        return ans

    def solve(self):
        trace = []

        walk, current = True, 1
        while walk:
            # Query possible neighbors
            trace.append(current)
            neighbors = self.islands[current]

            # Find island that's not go before
            current = next((x for x in neighbors if x not in trace), 0)
            walk = True if current else False

        print(' '.join(map(str, trace)))

if __name__ == '__main__':
    ans = Solution()
    ans.solve()
