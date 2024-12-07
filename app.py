from flask import Flask
from flask_cors import CORS
from flask import request, render_template
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text = True)
    data = json.loads(data)
    input_text = data['prompt']

    history_string = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors = "pt")
    outputs = model.generate(**inputs, max_length = 100)
    response = tokenizer.decode(outputs[0], skip_special_tokens = True).strip()

    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    app.run()