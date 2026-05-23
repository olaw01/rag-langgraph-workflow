import asyncio
import time
from typing import List

#---------------------------------------------------------------------------------------------------------------------
# Fake API calling - fake_api_call(5)/ fake_api_call(5, delay_s=2.5)
#---------------------------------------------------------------------------------------------------------------------

"""
Because fake_api_call() is an async function.
It doesn't return a "result", but a coroutine object, meaning a "task to be performed."
That's why it must be triggered by await.
However, await can only be used inside another async function.
example:
async def main():
    result = await fake_api_call(1)
    print(result)
asyncio.run(main())
"""

async def fake_api_call(i: int, delay_s: float =1.0) -> str:
    await asyncio.sleep(delay_s) # await - wait for the result of this operation, but do not block the entire program/ in real example: await client.get("https://api.example.com")
    return f"ok-{i}"



#---------------------------------------------------------------------------------------------------------------------
# SEQUENTIAL - a function that executes several queries one after the other (in line)
#---------------------------------------------------------------------------------------------------------------------

async def run_sequential(n: int) -> List[str]:

    results: List[str] = [] # WITHOUT HINT: results = []

    for i in range(n):
        results.append(await fake_api_call(i))
    return results



#---------------------------------------------------------------------------------------------------------------------
# GATHER - an unlimited parallel version that runs all queries at once.
#---------------------------------------------------------------------------------------------------------------------

async def run_parallel_unlimited(n: int) -> List[str]:
    tasks = [fake_api_call(i) for i in range(n)] #list comprehension tasks = [fake_api_call(0),...,fake_api_call(n-1)]
    return await asyncio.gather(*tasks) # *tasks - unpack a list [1, 2, 3] - 1 2 3



#---------------------------------------------------------------------------------------------------------------------
# SEMAPHORE - parallel version with limit (Do many tasks in parallel, but at a maximum limit at one time)
#---------------------------------------------------------------------------------------------------------------------

"""
Typical framework for async code with a concurrency limit:

sem = asyncio.Semaphore(limit)

async def wrapped(...):
    async with sem:
        return await any_async_function(...)

tasks = [wrapped(...) for ...]
return await asyncio.gather(*tasks)
"""

async def run_parallel_limited(n: int, limit: int) -> List[str]:

    sem = asyncio.Semaphore(limit) # up to {number of limit} functions can execute code inside "async with sem" simultaneously

    async def wrapped(i:int) -> str:
        async with sem:  #limit
            return await fake_api_call(i)

    tasks = [wrapped(i) for i in range(n)]
    return await asyncio.gather(*tasks) # tasks = [wrapped(0),...,wrapped(n-1)]


#---------------------------------------------------------------------------------------------------------------------
# ASYNC MAIN
#---------------------------------------------------------------------------------------------------------------------

async def main():
    n = 10

    print("=== SEQUENTIAL ===")
    t0 = time.perf_counter()
    results = await run_sequential(n)
    t1 = time.perf_counter()
    print(results)
    print(f"sequential: {t1 - t0:.2f}s")

    print("=== PARALLEL UNLIMITED ===")
    t0 = time.perf_counter()
    results = await run_parallel_unlimited(n)
    t1 = time.perf_counter()
    print(results)
    print(f"parallel unlimited: {t1 - t0:.2f}s")

    print("=== PARALLEL LIMITED ===")
    t0 = time.perf_counter()
    results = await run_parallel_limited(n, limit=3)
    t1 = time.perf_counter()
    print(results)
    print(f"parallel limited (3): {t1 - t0:.2f}s")




asyncio.run(main())