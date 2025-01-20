import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import { AutoTokenizer , pipeline, env, MarianTokenizer} from '@xenova/transformers';


const basePath = '/Users/johnazzinaro/Coding/versai-react-native/versai/models'
const en_to_romance = 'Helsinki-NLP/opus-mt-en-ROMANCE';
const romance_to_en = 'Helsinki-NLP/opus-mt-ROMANCE-en';
const feature = 'seq2seq-lm';
const path = 'Helsinki-NLP/opus-mt-en-ROMANCE-seq2seq-lm/model.onnx';

// const romanceTokenizer = new AutoTokenizer(en_to_romance);
// const enTokenizer = new AutoTokenizer(romance_to_en);


export default function App() {
  const [romanceEnTokenizer, setRomanceToEnTokenizer] = useState(null);
  const [enRomanceTokenizer, setEnToRomanceTokenizer] = useState(null);
  const [romanceEnTranslator, setRomanceToEnTranslator] = useState(null);
  const [enRomanceTranslator, setEnToRomanceTranslator] = useState(null);
  const [models_loaded, setModelsLoaded] = useState(false);

  useEffect(() => {
    const initTranslator = async () => {
      try {
        // Create custom tokenizer function
        const createCustomTokenizer = async (basePath, modelPath, feature) => {
          try {
            // Load vocab and config
            const vocab = await fetch(`${basePath}/${modelPath}-${feature}/tokenizer/vocab.json`).then(r => r.json());
            const config = await fetch(`${basePath}/${modelPath}-${feature}/tokenizer/tokenizer_config.json`).then(r => r.json());
            
            // Create reverse vocab for decoding
            const reverseVocab = Object.fromEntries(Object.entries(vocab).map(([k, v]) => [v, k]));
            
            // Create tokenizer function
            const tokenizer = async (text, options = {}) => {
              const MODEL_MAX_LENGTH = 128;  // Fixed length for this model
              
              const tokens = text.split(/\s+/);
              let input_ids = tokens.map(token => vocab[token] || vocab['<unk>']);
              
              // Truncate if too long
              if (input_ids.length > MODEL_MAX_LENGTH) {
                input_ids = input_ids.slice(0, MODEL_MAX_LENGTH);
              }
              
              // Pad if too short
              while (input_ids.length < MODEL_MAX_LENGTH) {
                input_ids.push(vocab['<pad>']);
              }
              
              // Create attention mask (1 for real tokens, 0 for padding)
              const attention_mask = new Array(MODEL_MAX_LENGTH).fill(0);
              for (let i = 0; i < Math.min(tokens.length, MODEL_MAX_LENGTH); i++) {
                attention_mask[i] = 1;
              }
              
              return {
                input_ids,
                attention_mask
              };
            };

            // Add properties to tokenizer
            tokenizer.decode = (tokens) => tokens.map(t => reverseVocab[t] || '<unk>').join(' ');
            tokenizer.vocab = vocab;
            tokenizer.reverseVocab = reverseVocab;
            tokenizer.config = config;
            tokenizer.model_max_length = 128;

            return tokenizer;
          } catch (error) {
            console.error('Error creating custom tokenizer:', error);
            throw error;
          }
        };

        // Create and test tokenizers
        try {
          const romance_to_en_tokenizer = await createCustomTokenizer(basePath, romance_to_en, feature);
          const en_to_romance_tokenizer = await createCustomTokenizer(basePath, en_to_romance, feature);

          // Test before setting in state
          const testResult = await romance_to_en_tokenizer("test text");

          // Only set in state if test passes
          if (testResult && testResult.input_ids) {
            setRomanceToEnTokenizer(() => romance_to_en_tokenizer);
            setEnToRomanceTokenizer(() => en_to_romance_tokenizer);
            console.log("Tokenizers set in state");
          } else {
            throw new Error("Tokenizer test failed");
          }

        } catch (error) {
          console.error("Tokenizer setup failed:", error);
        }

        // Initialize the translation pipeline
        console.log('Initializing translator');
        const romance_to_en_translator = await ort.InferenceSession.create(`${basePath}/${romance_to_en}-${feature}/model.onnx`);
        const en_to_romance_translator = await ort.InferenceSession.create(`${basePath}/${en_to_romance}-${feature}/model.onnx`);

        // Set the wrapped versions in state
        setRomanceToEnTranslator(romance_to_en_translator);
        setEnToRomanceTranslator(en_to_romance_translator);

        setModelsLoaded(true);

        console.log('Translator initialized');
      } catch (error) {
        console.error('Error initializing translator:', error);
      }
    };

    initTranslator();
  }, []);


  const translate = async (text, source_lang, target_lang) => {
    console.log( 'translating ', text, source_lang, target_lang);
    if (!models_loaded) {
      console.log('Translator not ready');
      return;
    }

    if (!text) {
      console.log('No text to translate');
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


    const inputs = await tokenizer(text);
    let decoder_input_ids = (await tokenizer("<pad>")).input_ids;
    const decoded_text = [];
    const max_length = 2;
    console.log('translator', translator);
    
    for (let i = 0; i < max_length; i++) {  
      console.log('i', i);
      try {
        const decoder_mask = new Array(decoder_input_ids.length).fill(1);

        const model_inputs = {
          input_ids: new ort.Tensor('int64', inputs.input_ids, [1, inputs.input_ids.length]),
          attention_mask: new ort.Tensor('int64', inputs.attention_mask, [1, inputs.attention_mask.length]),
          decoder_input_ids: new ort.Tensor('int64', decoder_input_ids, [1, decoder_input_ids.length]),
          decoder_attention_mask: new ort.Tensor('int64', decoder_mask, [1, decoder_mask.length])
        };
      
        const outputs = await translator.run(model_inputs);

        console.log('outputs', outputs);

        const lastLogits = outputs[0][0][outputs[0][0].length - 1];
        const predictedId = lastLogits.indexOf(Math.max(...lastLogits));
  
        console.log('predictedId', predictedId);
        
        decoded_text.push(predictedId);
        decoder_input_ids = [...decoder_input_ids, predictedId];
  
        if (predictedId === tokenizer.eos_token_id) {
          break;
        }
      } catch (error) {
        console.log('error', error);
      }


    }

    // Decode the output tokens to text
    const translated_text = await tokenizer.decode(decoded_text);
    console.log(`Input: ${text}`);
    console.log(`Translation: ${translated_text}`);
    
    return translated_text;
  }


  return (
    <View style={styles.container}>
      <Text>Hello World!</Text>
      <Button 
        title="Translate" 
        onPress={() => translate('Hello world', 'pt', 'en')}
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

