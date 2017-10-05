import requests
import time
import sys
import random
from urllib.parse import quote

url = "http://0.0.0.0:4001/ask/"

training_data = "../data/SimpleQuestions_v2/annotated_fb_data_train.txt"

errors = 0
correct = 0

with open(training_data) as f:
  for line in f:
    if random.random() < 0.1:
      (a, b, c, question) = line.split('\t')
      response = requests.get(url + quote(question))
      answer = response.content
      if "UNKNOWN" in bytes.decode(response.content):
        errors += 1
      else:
        correct += 1
        print(question)
        print(answer)
print("errors: ", errors)
print("correct: ", correct)
