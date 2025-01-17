import { StatusBar } from 'expo-status-bar';
import React, { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import * as ort from 'onnxruntime-react-native';
//import { AutoTokenizer } from '@huggingface/transformers'

const basePath = '/Users/johnazzinaro/Coding/versai-react-native/versai/models'
const en_to_romance = 'Helsinki-NLP/opus-mt-en-ROMANCE';
const romance_to_en = 'Helsinki-NLP/opus-mt-ROMANCE-en';
const feature = 'seq2seq-lm';
const path = 'Helsinki-NLP/opus-mt-en-ROMANCE-seq2seq-lm/model.onnx';

const romanceTokenizer = new AutoTokenizer(en_to_romance);
const enTokenizer = new AutoTokenizer(romance_to_en);


export default function App() {


  useEffect(() => {
    const createSession = async () => {
      try {
        const en_to_romance_session = await ort.InferenceSession.create(`${basePath}/${en_to_romance}-${feature}/model.onnx`);
        const romance_to_en_session = await ort.InferenceSession.create(`${basePath}/${romance_to_en}-${feature}/model.onnx`);
        

        console.log('Session created:');
      } catch (error) {
        console.error('Error creating session:', error );
      }
    };

    createSession();
  }, []);

  return (
    <View style={styles.container}>
      <Text>Hello World!</Text>
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
