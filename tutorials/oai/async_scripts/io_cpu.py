import asyncio
import time

# I/O密集型任务示例
async def io_intensive():
    print(f"开始I/O任务 {time.strftime('%H:%M:%S')}")
    await asyncio.sleep(2)  # 模拟I/O操作
    print(f"完成I/O任务 {time.strftime('%H:%M:%S')}")

# CPU密集型任务示例
async def cpu_intensive():
    print(f"开始CPU任务 {time.strftime('%H:%M:%S')}")
    for _ in range(50000000):  # 执行大量计算
        _ = _ ** 2
    print(f"完成CPU任务 {time.strftime('%H:%M:%S')}")

async def main():
    # I/O密集型任务并发执行
    print("测试I/O密集型任务:")
    await asyncio.gather(
        io_intensive(),
        io_intensive(),
        io_intensive()
    )  # 总耗时约2秒
    
    print("\n测试CPU密集型任务:")
    await asyncio.gather(
        cpu_intensive(),
        cpu_intensive(),
        cpu_intensive()
    )  # 总耗时是三个任务的总和

asyncio.run(main())