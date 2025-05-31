import asyncio
import aiohttp
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI

async def generate_text(prompt, client):
    response = await client.chat.completions.create(
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def process_batch(batch, client):
    responses = await asyncio.gather(*[
        client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        ) for prompt in batch
    ])
    return [response.choices[0].message.content for response in responses]
    
async def main():
    prompts = [f"Tell me a fact about number {i}" for i in range(100)]
    batch_size = 10
     
    async with AsyncOpenAI() as client:
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_results = await process_batch(batch, client)
            results.extend(batch_results)
     
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}\nResponse: {result}\n")
asyncio.run(main())


async def main():
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a haiku about artificial intelligence.",
        "Describe the process of photosynthesis."
    ]
     
    async with AsyncOpenAI() as client:
        tasks = [generate_text(prompt, client) for prompt in prompts]
        results = await asyncio.gather(*tasks)
     
    for prompt, result in zip(prompts, results):
        print(f"Prompt: {prompt}\nResponse: {result}\n")

if __name__ == '__main__':
    assert load_dotenv(find_dotenv())
    asyncio.run(main())