import random
import json

import torch
import time
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#from googlesearch import *
import webbrowser
#from pygame import mixer
import requests
from pycricbuzz import Cricbuzz
import billboard





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "LILY"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    #if tag=='google':
    #    query=input('Enter query')
    #     chrome_path = r'C:\Program Files\Google\Chrome\Application\chrome.exe %s'
    #    for url in search(query, tld="co.in", num=4, stop = 1, pause = 2):
    #        webbrowser.open("https://google.com/search?q=%s" % query)

    if tag=='datetime':        
        # print()
        print (time.strftime("%A"),time.strftime("%d %B %Y"),time.strftime("%H:%M:%S"))
        return (time.strftime("%A"),time.strftime("%d %B %Y"),time.strftime("%H:%M:%S"))


    

    if tag=='news':
        main_url = " http://newsapi.org/v2/top-headlines?country=in&apiKey=bc88c2e1ddd440d1be2cb0788d027ae2"
        open_news_page = requests.get(main_url).json()
        article = open_news_page["articles"]
        results = []

        for ar in article: 
            results.append([ar["title"],ar["url"]]) 
        return  results[1][0]+ results[1][1]
    
    if tag=='cricket':
         c = Cricbuzz()
         matches = c.matches()
         results=[]
         for match in matches:
             results.append(match['srs'])
         return results[1]

    if tag=='song':
        chart=billboard.ChartData('hot-100')
        print('The top 10 songs at the moment are:')
        results=[]
        for i in range(10):
            song=chart[i]
            results.append([song.title,song.artist])
        return results[1][0]+" ,"+results[2][0]+" ,"+results[3][0]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
bot_name = "LILY" 
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)





















