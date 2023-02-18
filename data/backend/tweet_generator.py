import openai
import pandas as pd
import os

class TweetGenerator:
    def __init__(self, tweet_topic, recent_tweets=None, successful_tweets=None, twitter_account_name="SMU", 
                 tweet_tone=None, prompt=None, model_name="text-davinci-003", num_responses=1, echo=False, 
                 temperature=0, top_p=1, max_tokens=50):

        self.openai = openai
        self.openai.api_key = os.environ['OPENAI_API_KEY']
        
        # Twitter information for generating the prompt
        self.tweet_topic = tweet_topic
        self.twitter_account_name = twitter_account_name
        self.recent_tweets = recent_tweets              # List of recent tweets to use as examples for GPT
        self.successful_tweets = successful_tweets      # List of successful tweets to use as examples for GPT
        self.tweet_tone = tweet_tone                    # Optional tone to write the tweet in
        self.prompt = prompt                            # We will generate a prompt if one is not passed
        
        # OpenAI API params
        self.model_name = model_name # One of: ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]
        self.max_tokens = max_tokens
        self.num_responses = num_responses
        self.echo = echo
        self.temperature = temperature
        self.top_p = top_p
        
        # Response from OpenAI API
        self.response = None
        self.generated_tweet = None
    
    def generate_tweet(self):
        
        self._set_prompt()

        print(f"=======\n{self.prompt}\n===========")
        
        self._request_model_response()
        
        return 
    
    def _request_model_response(self):
        
        self.response = self.openai.Completion.create(model=self.model_name, 
                                                 prompt=self.prompt, 
                                                 max_tokens=self.max_tokens, 
                                                 n=self.num_responses, 
                                                 echo=self.echo, 
                                                 temperature=self.temperature, 
                                                 top_p=self.top_p)
        
        self.generated_tweet = self.response['choices'][0]['text'].strip()
        
        return 
    
    def _set_prompt(self):
        
        if not self.prompt:
            recent_prompt = self._get_recent_tweets_prompt()
            
            successful_prompt = self._get_successful_tweets_prompt()
            
            tone_prompt = self._get_tone_prompt()
            
            if recent_prompt and successful_prompt:
                full_prompt = (f"{recent_prompt}{successful_prompt}"
                            f"The following is an {self.twitter_account_name} tweet about {self.tweet_topic}{tone_prompt} "
                            f"that is related to recent events and has a high chance of being successful and is less than 280 characters long:\n")
            elif successful_prompt:
                full_prompt = (f"{successful_prompt}"
                            f"The following is an {self.twitter_account_name} tweet about {self.tweet_topic}{tone_prompt} "
                            "that has a high chance of being successful and is less than 280 characters long:\n")
            elif recent_prompt:
                full_prompt = (f"{recent_prompt}"
                            f"The following is an {self.twitter_account_name} tweet about {self.tweet_topic}{tone_prompt} "
                            f"that is related to recent events and is less than 280 characters long:\n")
            
            self.prompt = full_prompt
        
        return 
        
    def _get_recent_tweets_prompt(self):
        
        if self.recent_tweets:
            prompt = f"The following are the most recent tweets written by {self.twitter_account_name}:\n"

            for index, tweet in enumerate(self.recent_tweets, start=1):
                prompt += f"{index}. {tweet}\n"

            prompt += "\n"
            
            return prompt
        else:
            return False
    
    def _get_successful_tweets_prompt(self):
        
        if self.successful_tweets:
            prompt = f"The following are the most successful tweets ever written by {self.twitter_account_name}:\n"

            for index, tweet in enumerate(self.successful_tweets, start=1):
                prompt += f"{index}. {tweet}\n"

            prompt += "\n"

            return prompt
        else:
            
            return False
        
    def _get_tone_prompt(self):
        
        if self.tweet_tone:
            return f", written in a {self.tweet_tone} tone,"
        else:
            return ""

