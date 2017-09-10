from flask import Flask
from server import Server

app = Flask(__name__)
server = Server()

@app.route('/api/v1.0/ask_question/<string:question>', methods=['GET'])
def answer_question(question):
    print(question)
    server.answer(question)
    print(answer)
    return answer

if __name__ == '__main__':
    server.setup()
    answer_question("where was sasha vujacic born?")
    app.run(debug=True)
