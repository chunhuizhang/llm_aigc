import asyncio
import openai
from openai import AsyncOpenAI
import backoff
from dotenv import load_dotenv, find_dotenv
import os

assert load_dotenv(find_dotenv())

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@backoff.on_exception(backoff.expo, openai.RateLimitError)
async def make_api_call_to_gpt(prompt, model="gpt-4o-mini"):
    messages = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    
    return response.choices[0].message.content

async def make_api_call_with_semaphore(prompt, semaphore, model="gpt-4o-mini"):
    async with semaphore:
        result = await make_api_call_to_gpt(prompt, model)
        return prompt, result

async def main():
    prompts = [
        "What model are you?",
        "How many Rs in strawberry?", 
        "Which is bigger, 0.9 or 0.11?",
        # 假设这里有更多的 prompts
    ]

    # 设置最大并发数，例如 5
    max_concurrent_requests = 5
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # 创建任务列表，包含信号量控制
    tasks = [
        make_api_call_with_semaphore(prompt, semaphore)
        for prompt in prompts
    ]

    # 使用 asyncio.as_completed 来实时处理完成的任务
    for coro in asyncio.as_completed(tasks):
        prompt, result = await coro
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")
        print('====================')

    # 如果需要等待所有任务完成，可以使用以下方法
    # results = await asyncio.gather(*tasks)
    # for prompt, result in results:
    #     print(prompt)
    #     print(result)
    #     print('====================')

if __name__ == "__main__":
    asyncio.run(main())
