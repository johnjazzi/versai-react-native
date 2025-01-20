
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

def test_model(model_name, text):
    feature = "seq2seq-lm"
    onnx_path = Path(f"models/{model_name}-{feature}/")
    tokenizer = AutoTokenizer.from_pretrained(f"versai/src/versai/models/{model_name}-{feature}/tokenizer")
    tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ROMANCE-en')
    ort_session = ort.InferenceSession(onnx_path / f"model.onnx")

    inputs = tokenizer(text, return_tensors="np", padding=True)


    max_length = 50
    decoded_text = []
    decoder_input_ids = tokenizer("<pad>", return_tensors="np")["input_ids"]

    for _ in range(max_length):
        # Prepare model inputs
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": np.ones_like(decoder_input_ids)  # Assuming all tokens are valid
        }
        
        outputs = ort_session.run(["logits"], model_inputs)
        predicted_ids = np.argmax(outputs[0], axis=-1)
        decoded_text.append(predicted_ids[0, -1])  # Get the last predicted token
        decoder_input_ids = np.concatenate([decoder_input_ids, predicted_ids[:, -1:]], axis=-1)

        if predicted_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    # Decode the output tokens to text
    translated_text = tokenizer.decode(decoded_text, skip_special_tokens=True)
    print(f"Input: {text}")
    print(f"Translation: {translated_text}")


# Main execution

test_model("Helsinki-NLP/opus-mt-ROMANCE-en",text="Hola, ¿cómo estás?")
test_model("Helsinki-NLP/opus-mt-en-ROMANCE",text=">>pt<< Hello, how are you?")

