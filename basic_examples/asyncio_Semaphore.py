import asyncio
import time

async def foo(name, delay, sem):
    async with sem:                 # LIMIT: ile takich foo może być naraz "w środku"
        print(time.strftime("%H:%M:%S"), "START", name)
        await asyncio.sleep(delay)  # udajemy czekanie na API
        print(time.strftime("%H:%M:%S"), "END  ", name)

async def main():

    sem = asyncio.Semaphore(2)      # zmień na 2 i zobacz różnicę

    await asyncio.gather(
        foo("A", 2, sem),
        foo("B", 2, sem),
        foo("C", 2, sem),
    )

asyncio.run(main())