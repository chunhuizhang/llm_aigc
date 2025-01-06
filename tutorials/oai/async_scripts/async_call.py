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

async def main():
    prompts = ["What model are you?",
               "How many Rs in strawberry?", 
               "Which is bigger, 0.9 or 0.11?"
              ]

    # Create a list to store the results of asynchronous calls
    results = []

    # Asynchronously call the function for each prompt
    tasks = [make_api_call_to_gpt(prompt) for prompt in prompts]
    print(tasks)
    results = await asyncio.gather(*tasks)
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(result)
        print('====================')
        
asyncio.run(main())
