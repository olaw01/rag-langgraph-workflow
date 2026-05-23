from __future__ import annotations

import asyncio
import time
from typing import List


async def fake_api_call(i: int, delay_s: float = 1.0) -> str:
    await asyncio.sleep(delay_s)
    return f"ok-{i}"


async def run_sequential(n: int) -> List[str]:
    results: List[str] = []
    for i in range(n):
        results.append(await fake_api_call(i))  # czekasz na 1, potem 2, potem 3...
    return results


async def run_parallel_unlimited(n: int) -> List[str]:
    tasks = [fake_api_call(i) for i in range(n)]  # przygotuj 10 "telefonów"
    return await asyncio.gather(*tasks)           # odpal je naraz


async def run_parallel_limited(n: int, limit: int) -> List[str]:
    sem = asyncio.Semaphore(limit)

    async def wrapped(i: int) -> str:
        async with sem:                           # max 'limit' na raz
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