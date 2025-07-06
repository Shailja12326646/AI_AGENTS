# load end
import os
from dotenv import load_dotenv
from openai import OpenAI


#load env

load_dotenv()
client=OpenAI()

#read task from file
def read_tasks(filepath):
    with open (filepath,"r") as f:
        return f.read()

def summarize_tasks(tasks):
    prompt= f"""
    You are a task planning agent.
    Given a list of tasks , categorize them into 3 priority
    buckets:
    -High Priority
    -Medium Priority
    -Low Priority
    Tasks:
    {tasks}
    Return the response in this format:
    High Priority:
    -task 1
    -task 2


    Medium Priority:
    -task 1
    -task 2

    Low Priority:
    -task 1
    -task 2



"""


    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user","content": prompt}
        ]
    )

    return response.choices[0].message.content


if __name__=="__main__":
    task_text=read_tasks("tasks.txt")
    summary=summarize_tasks(task_text)
    print("\n Task Summary")
    print("-"*30)
    print (summary)

# make a call to openAI  with prompts