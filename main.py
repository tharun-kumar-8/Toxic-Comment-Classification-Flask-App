from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

app = Flask(__name__)

# Load the saved model and tokenizer
model_name = "C:/Users/tharu/OneDrive/Desktop/MegaProjectNew/BertModel"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(
    model_name).to("cpu")  # Load model to CPU for inference

# Function to make predictions
def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer):
    user_input = [input_text]
 
    user_encodings = tokenizer(
        user_input, truncation=True, padding=True, return_tensors="pt")
 
    user_dataset = TensorDataset(
        user_encodings['input_ids'], user_encodings['attention_mask'])
 
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)
 
    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)
 
    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    predicted_labels = predicted_labels.tolist()
    labels_list = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']
    result = dict(zip(labels_list, predicted_labels[0]))
    return result

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def get_prediction():
    input_text = request.form["text"]
    result = predict_user_input(input_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
