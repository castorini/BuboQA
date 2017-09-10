from flask import Flask
from server import Server

app = Flask(__name__)
server = Server()

@app.route('/ask/<string:question>', methods=['GET'])
def answer_question(question):
    print(question)
    answer = server.answer(question)
    print(answer)
    return answer

if __name__ == '__main__':
    print("Setting up BuboQA")
    server.setup()
    #answer_question("where was sasha vujacic born?")
    print("BuboQA is up")
    app.run(port=4001)
