import asyncio

async def worker(name, semaphore):
    async with semaphore:
        print(f"{name} 获取到了信号量")
        await asyncio.sleep(2)  # 模拟一个需要时间的操作
        print(f"{name} 释放了信号量")

async def main():
    # 创建一个信号量，最多允许3个协程同时运行
    semaphore = asyncio.Semaphore(3)
    
    tasks = [worker(f"任务{i}", semaphore) for i in range(10)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
