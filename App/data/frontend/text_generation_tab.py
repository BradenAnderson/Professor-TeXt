import numpy as np
import pandas as pd

from bokeh.models import (Button, TabPanel, TextAreaInput, Paragraph, Dropdown, PlainText, Spinner,
TextInput, Div, Slider, Switch, CheckboxGroup, ScrollBox)
from bokeh.layouts import column, row

from ..backend.tweet_generator import TweetGenerator
from ..backend.tweet_retriever import TweetRetriever

class TextGenerationTab:
    def __init__(self):
        
        self.selections = None
        self.text_viz_df = None
        self.layout = None

        # List of most successful tweets to pass to tweet retriever
        self.most_successful_tweets = None 
        self.most_recent_tweets = None

        self.tweet_retriever = None
        self.tweet_generator = None

        # Current generated tweet (most probable if multiple were generated)
        self.generated_tweet = None

        # Controls
        self.button_generate_tweet = None 
        self.div_prompt_design_header = Div(text="<u><b>Prompt Design</b></u>", visible=True) 

        self.div_autogenerate_prompt = Div(text="Autogenerate Prompt:", visible=True)   
        self.switch_autogenerate_prompt = None
        
        self.div_twitter_account_name = Div(text="Twitter Account Name:", visible=True)  
        self.text_input_twitter_account_name = None
        
        self.div_tweet_topic = Div(text="Tweet Topic:", visible=True)  
        self.text_input_tweet_topic = None
        
        self.div_tweet_tone = Div(text="Tweet Tone:")
        self.dd_input_tweet_tone = None 

        self.div_use_recent_tweets = Div(text="Use Recent Tweets:")
        self.switch_use_recent_tweets = None 

        self.div_num_recent_tweets = Div(text="Number of Recent Tweets:")
        self.spinner_num_recent_examples = None 

        self.div_tweets_to_exclude = Div(text="Tweets Types to Exclude:")
        self.ms_tweets_to_exclude = None 

        self.div_use_success_examples = Div(text="Use Most Popular Tweets:")
        self.switch_use_success_examples = None 

        self.div_num_success_examples = Div(text="Number of Popular Tweets:")
        self.spinner_num_success_examples = None 

        self.div_success_metric = Div(text="Popularity Metric:")
        self.dd_success_metric = None 

        self.div_input_prompt = Div(text="Input Prompt Here:", visible=False)
        self.text_input_prompt = None 

        self.div_openai_header = Div(text="<u><b>OpenAI Model Settings</b></u>", visible=True) 
        
        self.div_openai_model = Div(text="OpenAI Model:")
        self.dd_openai_model = None 

        self.div_openai_temperature = Div(text="Model Temperature:")
        self.slider_openai_temperature = None

        self.div_openai_top_p = Div(text="Model top_p:")
        self.slider_openai_top_p = None 

        self.prompt_design_controls = None 
        self.openai_model_controls = None 

        self.div_prompt_used_header = Div(text="<u><b>Prompt Used</b></u>", visible=True) 
        self.prompt_used_paragraph = None 
        
        self.div_openai_max_tokens = Div(text="Model max_tokens:")
        self.spinner_openai_max_tokens = None

        self.div_tweet_generated_header = Div(text="<u><b>Tweet Generated</b></u>", visible=True) 
        self.tweet_generated_paragraph = None
        self.output_widgets = None 

        return


    def _initialize_prompt_design_controls(self):

        self.switch_autogenerate_prompt = Switch(active=True) 

        self.text_input_twitter_account_name = TextInput(title="", value=self.selections['text_gen_twitter_username'])

        self.text_input_tweet_topic = TextInput(title="", value='')

        self.dd_input_tweet_tone = Dropdown(label=f"tweet_tone={self.selections['text_gen_tweet_tone']}", 
                                            name="dd_tweet_tone", 
                                            menu=[("None", "None"), 
                                                  ("Serious", "serious"), 
                                                  ("Funny", "funny"), 
                                                  ("Casual", "casual")]) 
                                                  
        self.switch_use_recent_tweets= Switch(active=True) 
        
        self.spinner_num_recent_examples= Spinner(title="", 
                                                 name="spinner_num_recent_examples",
                                                 step=1, 
                                                 low=0, 
                                                 high=25,
                                                 value=self.selections['text_gen_num_recent_tweet_examples'])
        
        self.ms_tweets_to_exclude = CheckboxGroup(labels=["retweets", "replies"], active=[0, 1])
        
        self.switch_use_success_examples= Switch(active=True) 
        
        self.spinner_num_success_examples= Spinner(title="", 
                                                   name="spinner_num_success_examples",
                                                   step=1, 
                                                   low=0, 
                                                   high=25,
                                                   value=self.selections['text_gen_num_successful_tweet_examples'])
        
        self.dd_success_metric= Dropdown(label=f"metric={self.selections['text_gen_success_metric']}", 
                                         name="dd_tweet_tone", 
                                         menu=[self.selections['text_gen_success_metric']] +\
                                            [value for value in self.selections['numeric_metric_columns'] 
                                            if value != self.selections['text_gen_success_metric']])  
        
        self.text_input_prompt = TextAreaInput(title="", value="write your prompt here!", rows=15, cols=75, visible=False)

        self.prompt_design_controls = column(self.div_prompt_design_header,
                                             row(self.div_autogenerate_prompt, self.switch_autogenerate_prompt),
                                             column(self.div_twitter_account_name, self.text_input_twitter_account_name), 
                                             column(self.div_tweet_topic,self.text_input_tweet_topic),
                                             column(self.div_tweet_tone, self.dd_input_tweet_tone), 
                                             row(self.div_use_recent_tweets, self.switch_use_recent_tweets), 
                                             column(self.div_num_recent_tweets, self.spinner_num_recent_examples), 
                                             column(self.div_tweets_to_exclude, self.ms_tweets_to_exclude), 
                                             row(self.div_use_success_examples, self.switch_use_success_examples), 
                                             column(self.div_num_success_examples, self.spinner_num_success_examples), 
                                             column(self.div_success_metric, self.dd_success_metric), 
                                             column(self.div_input_prompt, self.text_input_prompt))  

        return 

    def _initialize_openai_model_controls(self):

        self.dd_openai_model = Dropdown(label=f"model={self.selections['text_gen_model_name']}", 
                                        name="dd_openai_model", 
                                        menu=[self.selections['text_gen_model_name']] +\
                                            [value for value in self.selections['text_gen_openai_models'] 
                                            if value != self.selections['text_gen_model_name']])  

        self.slider_openai_temperature = Slider(title="", 
                                                start=0.0, 
                                                end=2.0, 
                                                step=0.01, 
                                                format='0[.]00',
                                                value=self.selections['text_gen_temperature'])

        self.slider_openai_top_p = Slider(title="", 
                                          start=0.0, 
                                           end=1.0, 
                                           step=0.01, 
                                           format='0[.]00',
                                           value=self.selections['text_gen_top_p'])

        self.spinner_openai_max_tokens= Spinner(title="", 
                                                   name="spinner_max_tokens",
                                                   step=10, 
                                                   low=5, 
                                                   high=1_000,
                                                   value=self.selections['text_gen_max_tokens'])


        self.openai_model_controls = column(self.div_openai_header, 
                                            column(self.div_openai_model, self.dd_openai_model), 
                                            column(self.div_openai_temperature, self.slider_openai_temperature), 
                                            column(self.div_openai_top_p, self.slider_openai_top_p), 
                                            column(self.div_openai_max_tokens, self.spinner_openai_max_tokens)) 

    def _initialize_output_widgets(self):
        self.prompt_used_paragraph = Div(text='')
        self.prompt_used_paragraph_scrollable = ScrollBox(child=self.prompt_used_paragraph, height=400, width=500, sizing_mode="fixed")

        self.tweet_generated_paragraph = Div(text='')
        self.tweet_generated_paragraph_scrollable = ScrollBox(child=self.tweet_generated_paragraph, height=400, width=500, sizing_mode="fixed")                               

        self.output_widgets = column(column(self.div_prompt_used_header, self.prompt_used_paragraph_scrollable), 
                                     column(self.div_tweet_generated_header, self.tweet_generated_paragraph_scrollable))

        return 

    def _initialize_controls(self):

        self.button_generate_tweet = Button(label="Generate Tweet", 
                                            button_type="success", 
                                            sizing_mode="stretch_width",
                                            aspect_ratio="auto", 
                                            max_height=40, 
                                            max_width=500)

        self._initialize_prompt_design_controls()
        self._initialize_openai_model_controls()
        self._initialize_output_widgets()

        return

    def setup(self):

        self._initialize_controls()

        self.set_most_successful_tweets()

        return 

    def update(self):

        self.set_most_successful_tweets()

        self.run_tweet_generation_pipeline()

        # TODO: Update widget to display the prompt that was used!

    def _set_tab_layout(self):

        # layout = row(column(self.prompt_design_controls, self.openai_model_controls), 
        #                   self.output_widgets)

        layout = row(column(row(self.prompt_design_controls, 
                                self.openai_model_controls), 
                            self.button_generate_tweet), 
                     self.output_widgets)
                     
        self.layout = TabPanel(child=layout, title="Tweet Generation")

        return

    def get_layout(self):

        self._set_tab_layout()

        return self.layout 

    def run_tweet_generation_pipeline(self):
        
        # Don't hit twitter api unless we changed something that would change its results
        if self.selections['text_gen_run_tweet_retriever'] and self.selections['text_gen_use_recent_tweets']:
            self._run_tweet_retriever()

        # Don't include recent tweets section in autogenerated prompt
        if not self.selections['text_gen_use_recent_tweets']:
            self.most_recent_tweets = None

        self._run_tweet_generator()

        return 
    
    def set_most_successful_tweets(self):
        
        if self.selections['text_gen_use_success_tweets']:
            self.most_successful_tweets = self.text_viz_df.sort_values(by=self.selections['text_gen_success_metric'], 
                                                                    ascending=self.selections['text_gen_success_metric_bigger_is_better'])[self.selections['text_column']].tolist()
            
            self.most_successful_tweets = self.most_successful_tweets[:self.selections['text_gen_num_successful_tweet_examples']]
        else:
            self.most_successful_tweets = None 

        return

    def _run_tweet_generator(self):

        self.tweet_generator = TweetGenerator(recent_tweets=self.most_recent_tweets, 
                                              successful_tweets=self.most_successful_tweets, 
                                              tweet_topic=self.selections['text_gen_tweet_topic'], 
                                              twitter_account_name=self.selections['text_gen_twitter_username'], 
                                              tweet_tone=self.selections['text_gen_tweet_tone'], 
                                              model_name=self.selections['text_gen_model_name'], 
                                              num_responses=self.selections['text_gen_num_tweets_to_generate'],  
                                              temperature=self.selections['text_gen_temperature'], 
                                              prompt=self.selections['text_gen_prompt'],
                                              top_p=self.selections['text_gen_top_p'], 
                                              echo=False,
                                              max_tokens=self.selections['text_gen_max_tokens'])

        self.tweet_generator.generate_tweet()
        
        self.generated_tweet = self.tweet_generator.generated_tweet

        return 

    def _run_tweet_retriever(self):

        self.tweet_retriever = TweetRetriever(username=self.selections['text_gen_twitter_username'], 
                                              max_tweets=max(self.selections['text_gen_num_recent_tweet_examples'], 5), 
                                              tweets_to_exclude=self.selections['text_gen_tweets_to_exclude'])
        self.tweet_retriever.run()

        self.most_recent_tweets = self.tweet_retriever.tweet_df['Tweet'].tolist()[:self.selections['text_gen_num_recent_tweet_examples']]

        return

    def set_selections(self, selections):
        self.selections = selections
        return

    def set_text_viz_df(self, text_viz_df):
        self.text_viz_df = text_viz_df
        return