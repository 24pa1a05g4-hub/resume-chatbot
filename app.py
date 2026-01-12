from flask import Flask, render_template, request, jsonify
from rag import get_answer

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    bot_response = get_answer(user_message)
    return jsonify({"answer": bot_response})

if __name__ == "__main__":
    app.run(debug=False)