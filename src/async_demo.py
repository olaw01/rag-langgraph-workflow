from __future__ import annotations

import asyncio
import time
from typing import List


async def fake_api_call(i: int, delay_s: float = 1.0) -> str:
    """Udaje request do API: czeka delay_s sekund i zwraca wynik."""
    await asyncio.sleep(delay_s)
    return f"ok-{i}"


async def run_sequential(n: int) -> List[str]:
    """N wywołań jedno po drugim (wolno)."""
    results: List[str] = []
    for i in range(n):
        results.append(await fake_api_call(i))
    return results


async def run_parallel_unlimited(n: int) -> List[str]:
    """N wywołań naraz (szybko, ale bez limitu)."""
    tasks = [fake_api_call(i) for i in range(n)]
    return await asyncio.gather(*tasks)


async def run_parallel_limited(n: int, limit: int) -> List[str]:
    """N wywołań naraz, ale max 'limit' równocześnie (produkcyjnie)."""
    sem = asyncio.Semaphore(limit)

    async def wrapped(i: int) -> str:
        async with sem:
            return await fake_api_call(i)

    tasks = [wrapped(i) for i in range(n)]
    return await asyncio.gather(*tasks)


async def main() -> None:
    n = 10

    t0 = time.perf_counter()
    await run_sequential(n)
    t1 = time.perf_counter()
    print(f"sequential: {t1 - t0:.2f}s")

    t0 = time.perf_counter()
    await run_parallel_unlimited(n)
    t1 = time.perf_counter()
    print(f"parallel unlimited: {t1 - t0:.2f}s")

    t0 = time.perf_counter()
    await run_parallel_limited(n, limit=3)
    t1 = time.perf_counter()
    print(f"parallel limited (3): {t1 - t0:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())