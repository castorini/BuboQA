import requests
import sys
from urllib.parse import quote


url = "http://0.0.0.0:4001/ask/"

question = quote(sys.argv[1])
print(question)
url += question

response = requests.get(url)
print(response.content)
