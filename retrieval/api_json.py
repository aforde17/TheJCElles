import json
import tweepy
from tweepy import OAuthHandler

# Authenticate
CONSUMER_KEY = "" #@param {type:"string"}
CONSUMER_SECRET_KEY = "" #@param {type:"string"}
ACCESS_TOKEN_KEY = "" #@param {type:"string"}
ACCESS_TOKEN_SECRET_KEY = "" #@param {type:"string"}

#Creates a JSON Files with the API credentials
with open('api_keys.json', 'w') as outfile:
    json.dump({
    "consumer_key":CONSUMER_KEY,
    "consumer_secret":CONSUMER_SECRET_KEY,
    "access_token":ACCESS_TOKEN_KEY,
    "access_token_secret": ACCESS_TOKEN_SECRET_KEY
     }, outfile)
