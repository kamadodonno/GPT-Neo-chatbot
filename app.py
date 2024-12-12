from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize app
app = Flask(__name__)

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")  # Use "cuda" if on GPU

@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.json
    input_text = data.get("input", "")
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
