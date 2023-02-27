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
        trace = [1]

        current = 1
        while True:
            # Query possible neighbors, sort to guarantee ascending order
            ref = self.islands[current]
            # ref.sort()

            # Find island that's not go before
            found = False
            for adjacency in ref:
                if adjacency in trace:
                    continue

                found = True
                trace.append(adjacency)
                current = adjacency
                break

            if not found:
                break

        print(' '.join(map(str, trace)))

if __name__ == '__main__':
    ans = Solution()
    ans.solve()
