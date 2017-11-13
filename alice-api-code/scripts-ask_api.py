import requests
import sys
from urllib.parse import quote


url = "0.0.0.0:4001/ask/"
question = quote(sys.argv[1])
url += question

print(requests.get(url))
