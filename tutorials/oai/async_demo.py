import asyncio
import time

async def make_coffee():
    print("开始煮咖啡...")
    await asyncio.sleep(3)  # 模拟煮咖啡的时间
    print("咖啡准备好了！")
    return "一杯咖啡"

async def make_toast():
    print("开始烤面包...")
    await asyncio.sleep(2)  # 模拟烤面包的时间
    print("面包烤好了！")
    return "一片吐司"

async def main():
    # 同时开始准备咖啡和面包
    t0 = time.time()
    coffee_task = asyncio.create_task(make_coffee())
    toast_task = asyncio.create_task(make_toast())
    
    # 等待两个任务都完成
    coffee, toast = await asyncio.gather(coffee_task, toast_task)
    print(f"早餐准备完成：{coffee}和{toast}, 用时: {time.time() - t0:.2f}s")

# 运行程序
asyncio.run(main())