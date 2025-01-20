

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.marian import MarianOnnxConfig
from pathlib import Path
import numpy as np

#TODO: Send this directly to the versai/scr/ dir


# Export model to ONNX format
def export_and_test_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    feature = "seq2seq-lm"
    onnx_path = Path(f"versai/src/versai/models/{model_name}-{feature}/")
    onnx_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Run ONNX conversion directly using the export function
    from transformers.onnx import export
    export(
        preprocessor=tokenizer,
        model=model,
        config=MarianOnnxConfig(model.config, task=feature),
        opset=12,
        output=onnx_path / f"model.onnx"
    )


    # Test the exported model
    batch_size = 1
    encoder_inputs = tokenizer(
        ["Hola, ¿cómo estás?"] * batch_size,
        return_tensors="np",
    )
    decoder_inputs = tokenizer(
        ["Hello, how are you?"] * batch_size,
        return_tensors="np",
    )

    # Prepare all inputs for testing
    all_inputs = {
        "input_ids": encoder_inputs["input_ids"],
        "attention_mask": encoder_inputs["attention_mask"],
        "decoder_input_ids": decoder_inputs["input_ids"],
        "decoder_attention_mask": decoder_inputs["attention_mask"],
    }

    # Initialize ONNX runtime session and test the model

    import onnxruntime as ort
    ort_session = ort.InferenceSession(f"{onnx_path}/model.onnx")
    onnx_config = MarianOnnxConfig(model.config, task=feature)
    onnx_named_outputs = list(onnx_config.outputs.keys())
    onnx_outputs = ort_session.run(onnx_named_outputs, all_inputs)

    print(f"ONNX model test completed successfully for {model_name}!")

    # Save tokenizer
    tokenizer.save_pretrained(f"versai/src/versai/models/{model_name}-{feature}/tokenizer")

# Main execution
model_names = [
    "Helsinki-NLP/opus-mt-ROMANCE-en",
    "Helsinki-NLP/opus-mt-en-ROMANCE"
]

for model_name in model_names:
    export_and_test_model(model_name)