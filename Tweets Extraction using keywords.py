
import tweepy
import csv

#Twitter API credentials
consumer_key = 'R28pia41moOqdKkEJxLZISjY0'
consumer_secret = '3GvFT8W2aIT02zI3iymowgMuo99fPknz0tJMaZhoZGg3t4h1Zs'
access_token = '1028591573510967296-V8LqzzyaKHOdf3LRK2zp3oMpTJZ5Bn'
access_token_secret = 'odmldpUuPtJM9SxRt0g0jaYy642KxgzBPq5aWKx04Zb2o'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('mumbai.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="@Uber_BLR",
                           lang="en",
                           since="2018-04-03").items():
    print (tweet.created_at, tweet.text)
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])