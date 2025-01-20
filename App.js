import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import { Platform } from 'react-native';
import * as ortNative from 'onnxruntime-react-native';
import { AutoTokenizer } from '@xenova/transformers';

// Initialize web ONNX runtime
const initWebOrt = () => {
  return new Promise((resolve) => {
    if (Platform.OS !== 'web') {
      resolve(ortNative);
      return;
    }

    // Check if already loaded
    if (window.ort) {
      resolve(window.ort);
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js';
    script.async = true;
    
    script.onload = () => {
      console.log('ONNX Runtime Web loaded');
      if (window.ort) {
        if (window.ort.env && window.ort.env.wasm) {
          window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/';
        }
        resolve(window.ort);
      }
    };
    
    script.onerror = (err) => {
      console.error('Error loading ONNX Runtime Web:', err);
      resolve(null);
    };

    document.body.appendChild(script);
  });
};
const basePath = '/Users/johnazzinaro/Coding/versai-react-native/versai/models'
//const basePath = '/assets/models';
const en_to_romance = 'Helsinki-NLP/opus-mt-en-ROMANCE';
const romance_to_en = 'Helsinki-NLP/opus-mt-ROMANCE-en';
const feature = 'seq2seq-lm';

// const romanceTokenizer = new AutoTokenizer(en_to_romance);
// const enTokenizer = new AutoTokenizer(romance_to_en);


export default function App() {
  const [romanceEnTokenizer, setRomanceToEnTokenizer] = useState(null);
  const [enRomanceTokenizer, setEnToRomanceTokenizer] = useState(null);
  const [romanceEnTranslator, setRomanceToEnTranslator] = useState(null);
  const [enRomanceTranslator, setEnToRomanceTranslator] = useState(null);
  const [models_loaded, setModelsLoaded] = useState(false);
  const [ort, setOrt] = useState(null);

  useEffect(() => {
    const initTranslator = async () => {
      try {
        console.log('Initializing ONNX Runtime...');

        const ort = await initWebOrt();
        setOrt(() => ort);
        
        if (!ort) {throw new Error('Failed to initialize ONNX Runtime');}
        
        try {
          //const romance_to_en_tokenizer = await createCustomTokenizer(basePath, romance_to_en, feature);
          //const en_to_romance_tokenizer = await createCustomTokenizer(basePath, en_to_romance, feature);
          
          const romance_to_en_tokenizer = await AutoTokenizer.from_pretrained('xenova/opus-mt-ROMANCE-en');
          const en_to_romance_tokenizer = await AutoTokenizer.from_pretrained('xenova/opus-mt-en-ROMANCE');

          // romance_to_en_tokenizer.save_pretrained('models/xenova/opus-mt-ROMANCE-en');
          // en_to_romance_tokenizer.save_pretrained('en_to_romance_tokenizer');

          setRomanceToEnTokenizer(() => romance_to_en_tokenizer);
          setEnToRomanceTokenizer(() => en_to_romance_tokenizer);
          
          console.log('Tokenizers created successfully');
        } catch (error) {
          console.error("Tokenizer setup failed:", error);
          throw error;
        }

        console.log('Creating inference sessions...');
        const romance_to_en_translator = await ort.InferenceSession.create(
          `${basePath}/${romance_to_en}-${feature}/model.onnx`
        );
        const en_to_romance_translator = await ort.InferenceSession.create(
          `${basePath}/${en_to_romance}-${feature}/model.onnx`
        );

        setRomanceToEnTranslator(() => romance_to_en_translator);
        setEnToRomanceTranslator(() => en_to_romance_translator);
        setModelsLoaded(true);

        console.log('Translator initialized successfully');
      } catch (error) {
        console.error('Error initializing translator:', error);
        console.error('Error details:', error.stack);
      }
    };

    initTranslator();
  }, []);



  // Make sure to handle null text in your tokenizer
  const createCustomTokenizer = async (basePath, modelPath, feature) => {
    try {
      console.log(`Loading tokenizer from ${basePath}/${modelPath}-${feature}/tokenizer/vocab.json`);
      const vocab = await fetch(`${basePath}/${modelPath}-${feature}/tokenizer/vocab.json`).then(r => r.json());
      const config = await fetch(`${basePath}/${modelPath}-${feature}/tokenizer/tokenizer_config.json`).then(r => r.json());
      
      const reverseVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
      
      // Define the tokenizer function
      const tokenizer = (text, options = {}) => {
        const { pad = false } = options; // Destructure pad option
        if (!text) {
          console.warn('Received empty text for tokenization');
          return { input_ids: [], attention_mask: [] }; // Handle null or empty text
        }
        
        const MODEL_MAX_LENGTH = 128;
        // Adjust tokenization to handle punctuation correctly
        const tokens = text.match(/\S+|\s+/g) || []; // Split by whitespace and keep punctuation
        let input_ids = tokens.map(token => vocab[token.trim()] || vocab['<unk>']); // Trim tokens to avoid leading/trailing spaces
        
        if (input_ids.length > MODEL_MAX_LENGTH) {
          input_ids = input_ids.slice(0, MODEL_MAX_LENGTH);
        }
        
        if (pad) { // Check if padding is required
          while (input_ids.length < MODEL_MAX_LENGTH) {
            input_ids.push(vocab['<pad>']);
          }
        }
        
        // Adjust attention_mask to match input_ids length without padding
        const attention_mask = new Array(input_ids.length).fill(0);
        for (let i = 0; i < input_ids.length; i++) {
          attention_mask[i] = 1;
        }
        
        return { input_ids, attention_mask };
      };

      // Add a decode method to the tokenizer
      tokenizer.decode = (input_ids) => {
        return input_ids.map(id => reverseVocab[id] || '<unk>').join(' ');
      };

      console.log('Tokenizer function created successfully');
      return tokenizer; // Ensure the tokenizer function is returned
    } catch (error) {
      console.error('Error creating custom tokenizer:', error);
      throw error;
    }
  };

  const argmax = (array) => {
    return array.reduce((maxIndex, currentValue, currentIndex, arr) => {
      return currentValue > arr[maxIndex] ? currentIndex : maxIndex;
    }, 0);
  };

  const translate = async (text, source_lang, target_lang) => {
    console.log( 'translating ', text, source_lang, target_lang);
    if (!models_loaded) {
      console.log('Translator not ready');
      return;
    }

    let tokenizer, translator;

    if (source_lang === 'en') {
      tokenizer = enRomanceTokenizer;
      translator = enRomanceTranslator;
      text = `>>${target_lang}<< ${text}`
    } else {
      tokenizer = romanceEnTokenizer;
      translator = romanceEnTranslator;
    }

    const inputs = await tokenizer(text, {return_tensors: 'np'});
    let decoder_inputs= (await tokenizer("<pad>", {return_tensors: 'np'}));
  
    const max_length = 15;

    console.log()

    for (let i = 0; i < max_length; i++) {  
      try {

        const model_inputs = {
          input_ids: inputs.input_ids,
          attention_mask: inputs.attention_mask,
          decoder_input_ids: decoder_inputs.input_ids,
          decoder_attention_mask: decoder_inputs.attention_mask
        };

        console.log(model_inputs)

        const outputs = await translator.run(model_inputs);
        const logits_array = outputs.logits.cpuData;
        const predictedId = argmax(logits_array);

        console.log('predictedId', predictedId);
        
        const newDecoderInputIdIndex = decoder_inputs.decoder_input_ids.size; // Get current size for decoder_input_ids
        decoder_inputs.input_ids.data[newDecoderInputIdIndex] = predictedId; // Append predictedId
        decoder_inputs.input_ids.size += 1; // Increment size

        const newDecoderAttentionMaskIndex = decoder_inputs.decoder_attention_mask.size; // Get current size for decoder_attention_mask
        decoder_inputs.attention_mask.data[newDecoderAttentionMaskIndex] = 1; // Append 1 to attention mask
        decoder_inputs.attention_mask.size += 1; // Increment size
  
        if (predictedId === tokenizer.eos_token_id) {
          break;
        }
      } catch (error) {
        console.log('error', error);
      }


    }

    // Decode the output tokens to text
    const translated_text = await tokenizer.decode(decoder_input_ids);
    console.log(`Input: ${text}`);
    console.log(`Translation: ${translated_text}`);
    
    return translated_text;
  }


  return (
    <View style={styles.container}>
      <Text>Hello World!</Text>
      <Button 
        title="Translate" 
        onPress={() => translate(" Hello, how are you?", 'en', 'pt')}
      />
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

