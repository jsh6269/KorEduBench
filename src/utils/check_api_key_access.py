from typing import List

import openai
import requests
from tqdm import tqdm


def openai_entry(key_lst: List[str], model: str = "text-davinci-003"):
    access_key_lst = []
    unaccess_key_lst = []
    prompt = "Please write a tagline for an ice-cream shop."
    # prompt = "i like cats."
    # prompt = "INPUT: A fun family movie that 's suitable for all ages -- a movie that will make you laugh , cry and realize , ` It 's never too late to believe in your dreams . '\nSENTIMENT: Positive\n\nExplain the reasoning process for determining the overall SENTIMENT of the INPUT (limit to 150 tokens).\n\n"
    for key_item in tqdm(key_lst):
        openai.api_key = key_item
        try:
            response_lst = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=0,
                max_tokens=256,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=0,
            )
            access_key_lst.append(key_item)
            print(response_lst)
        except openai.error.RateLimitError as limiterror:
            limiterror_info = str(limiterror)
            if (
                limiterror_info
                == "You exceeded your current quota, please check your plan and billing details."
            ):
                unaccess_key_lst.append(key_item)
            print(model, "openai.error.RateLimitError", limiterror)
        except openai.error.InvalidRequestError as e:
            print(model, "openai.error.InvalidRequestError", e)
            unaccess_key_lst.append(key_item)
        except KeyboardInterrupt:
            break
        except:
            unaccess_key_lst.append(key_item)

    print("=" * 30)
    print(f"Total len {len(key_lst)}")
    print(f"Access key {len(access_key_lst)}")
    print(f"Unaccess key {len(unaccess_key_lst)}")
    print("=" * 30)
    for key in access_key_lst:
        print(f'"{key}",')


if __name__ == "__main__":
    # Example usage:
    # key_lst = [
    #     "sk-YOUR_OPENAI_API_KEY_HERE",
    # ]
    # openai_entry(key_lst, model="text-davinci-003")

    # Load API key from environment variable
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        key_lst = [api_key]
        openai_entry(key_lst, model="text-davinci-003")
    else:
        print("Warning: OPENAI_API_KEY environment variable not set")
