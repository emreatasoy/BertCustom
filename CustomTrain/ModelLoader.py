import json
import flask
from flask import request, jsonify
from flask_cors import CORS

import torch
from transformers import DistilBertTokenizerFast

def query_text(text) :
    encodings = tokenizer(text, return_tensors='pt')  # Bu tokenizer train yapılan tokenizer ile aynı olmalı
    input_ids = encodings["input_ids"]

    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits
    max_index = torch.argmax(logits)
    max_index += 1
    with open('mapping.json') as json_file:
        labels_of_indexes = json.load(json_file)
    keys = [k for k, v in labels_of_indexes.items() if v == max_index]

    print("query text : " + str(text) + " and intention: " + str(keys))
    return keys

torch.zeros(1).cuda()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cuda_available = torch.cuda.is_available()

model_path: str = './atis_trained_model.pt'

text = "I run to Ankara"

# Load
model = torch.load(model_path)

model.to(device)

model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

intention = query_text(text)
print(intention)


app = flask.Flask(__name__)
CORS(app)
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.
books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '199"2'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]


@app.route('/api/query/', methods=['POST'])
def api_query_text():
    print('server requested')
    query_parameters = request.get_json()
    results = []
    text = query_parameters.get('queryText').get('content')
    print("query param : " + str(query_parameters))
    print("query text : " + str(text))
    #text = query_parameters.get('text')

    intentionResList = query_text(text)
    for intention in intentionResList:
        res = {'intention': intention}
        results.append(res)


    resJson = jsonify(results)
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return resJson;

app.run()