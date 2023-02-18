import tweepy
import os
import pandas as pd

class TweetRetriever:
    def __init__(self, username="SMU", max_tweets=10, tweets_to_exclude=["retweets", "replies"]):
        
        self.max_tweets = max_tweets # max allowed value is 100
        self.tweets_to_exclude = tweets_to_exclude
        
        self.client = tweepy.Client(bearer_token=os.environ['TWITTER_API_BEARER_TOKEN'],
                                    consumer_key=os.environ['TWITTER_API_KEY'], 
                                    consumer_secret=os.environ['TWITTER_API_SECRET'], 
                                    access_token=os.environ['TWITTER_ACCESS_TOKEN'], 
                                    access_token_secret=os.environ['TWITTER_ACCESS_TOKEN_SECRET'])
        
        self.username = username
        self.user_id = self._get_user_id()
        
        self.tweet_response = None
        self.tweets = None
        self.tweet_ids = None
        self.tweet_df = None
    
    def run(self):
        
        self._get_tweets()
        
        return
    
    def _get_tweets(self):
        
        self.tweet_response = self.client.get_users_tweets(id=self.user_id, 
                                                           max_results=self.max_tweets, 
                                                           exclude=self.tweets_to_exclude, 
                                                           user_auth=True)
        
        self.tweets = [tweet.text for tweet in self.tweet_response.data]
        self.tweet_ids = [tweet.id for tweet in self.tweet_response.data]
        
        self.tweet_df = pd.DataFrame({'Tweet':self.tweets, 
                                      'Tweet ID':self.tweet_ids})
    
    def _get_user_id(self):
        response = self.client.get_user(username=self.username, user_auth=True)
        return response.data.id
