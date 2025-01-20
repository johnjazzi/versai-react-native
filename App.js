import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Button } from 'react-native';
import * as ort from 'onnxruntime-react-native';
import { AutoTokenizer } from '@xenova/transformers';
//import { AutoTokenizer } from '@huggingface/transformers'





const basePath = '/Users/johnazzinaro/Coding/versai-react-native/versai/models'
const en_to_romance = 'Helsinki-NLP/opus-mt-en-ROMANCE';
const romance_to_en = 'Helsinki-NLP/opus-mt-ROMANCE-en';
const feature = 'seq2seq-lm';
const path = 'Helsinki-NLP/opus-mt-en-ROMANCE-seq2seq-lm/model.onnx';

// const romanceTokenizer = new AutoTokenizer(en_to_romance);
// const enTokenizer = new AutoTokenizer(romance_to_en);


export default function App() {
  const [translator, setTranslator] = useState(null);
  
  useEffect(() => {
    const initTranslator = async () => {
      try {
        // Initialize the translation pipeline
        const romance_to_en_translator = await ort.InferenceSession.create(`${basePath}/${romance_to_en}-${feature}/model.onnx`);
        const en_to_romance_translator = await ort.InferenceSession.create(`${basePath}/${en_to_romance}-${feature}/model.onnx`);
        const romance_to_en_tokenizer = await AutoTokenizer.from_pretrained(`${basePath}/${romance_to_en}-${feature}/tokenizer`);
        const en_to_romance_tokenizer = await AutoTokenizer.from_pretrained(`${basePath}/${en_to_romance}-${feature}/tokenizer`);
        console.log('Translator initialized');
      } catch (error) {
        console.error('Error initializing translator:', error);
      }
    };

    initTranslator();
  }, []);


  const handleTranslate = async () => {
    if (!translator) {
      console.log('Translator not ready');
      return;
    }
    
    try {
      const result = await translator('Hello world');
      console.log('Translation result:', result);
    } catch (error) {
      console.error('Translation error:', error);
    }
  };

  return (
    <View>
      <Text>Hello World!</Text>
      <Button 
        title="Translate" 
        onPress={handleTranslate}
      />
      <StatusBar style="auto" />
    </View>
  );
}

