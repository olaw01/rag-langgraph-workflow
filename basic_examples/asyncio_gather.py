import time
import asyncio

async def foo(text, delay):
# asynchroniczny sleep bo wcześniejszy był jednowątkowy
    await asyncio.sleep(delay)

    print(text)

async def main():

    print("Started at time: ", time.strftime("%I:%M:%S"))

    await asyncio.gather(foo("Hello", delay=1), foo("World", delay=2))

    print("Ended at time: ", time.strftime("%I:%M:%S"))


asyncio.run(main())