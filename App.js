import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useState } from 'react';
import { StyleSheet, Text, View, Button, TextInput, TouchableOpacity, SafeAreaView } from 'react-native';
import { Platform } from 'react-native';
import * as ortNative from 'onnxruntime-react-native';
//import { AutoTokenizer , pipeline} from '@xenova/transformers';
import { Ionicons } from '@expo/vector-icons';
import { pipeline } from '@huggingface/transformers';


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


const initTransformersJS = () => {
  return new Promise((resolve) => {
    // Check if already loaded
    if (window.transformers) {
      resolve(window.transformers);
      return;
    }

    try {
      const script = document.createElement('script');
      script.type = 'module';
      script.src = 'https://unpkg.com/@huggingface/transformers';
      script.async = true;
      
      script.onload = () => {
        console.log('Transformers.js loaded');
        console.log(window);
        if (window.transformers) {
          console.log(window.transformers);
          resolve(window.transformers);
          return;
        }
      };
    
      script.onerror = (err) => {
        console.error('Error loading Transformers.js:', err);
        resolve(null);
      };

      document.body.appendChild(script);
    } catch (error) {
      console.error('Error loading Transformers.js:', error);
      resolve(null);
    }
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
  const [transformers, setTransformers] = useState(null);
  const [sourceText, setSourceText] = useState('');
  const [targetText, setTargetText] = useState('');
  const [sourceLang, setSourceLang] = useState('en');
  const [targetLang, setTargetLang] = useState('es');

  useEffect(() => {
    const initTranslator = async () => {
      try {
        setModelsLoaded(false);
        console.log('Initializing ONNX Runtime...');

        const ort = await initWebOrt();
        setOrt(() => ort);

        
        if (!ort) {throw new Error('Failed to initialize ONNX Runtime');}
   //     if (!transformers) {throw new Error('Failed to initialize Transformers');}

        
        // try {
        //   const romance_to_en_tokenizer = await AutoTokenizer.from_pretrained('xenova/opus-mt-ROMANCE-en');
        //   const en_to_romance_tokenizer = await AutoTokenizer.from_pretrained('xenova/opus-mt-en-ROMANCE');
        //   setRomanceToEnTokenizer(() => romance_to_en_tokenizer);
        //   setEnToRomanceTokenizer(() => en_to_romance_tokenizer);
          
        //   console.log('Tokenizers created successfully');
        // } catch (error) {
        //   console.error("Tokenizer setup failed:", error);
        //   throw error;
        // }

        // console.log('Creating inference sessions...');
        // const romance_to_en_translator = await ort.InferenceSession.create(
        //   `${basePath}/${romance_to_en}-${feature}/model.onnx`
        // );
        // const en_to_romance_translator = await ort.InferenceSession.create(
        //   `${basePath}/${en_to_romance}-${feature}/model.onnx`
        // );

        console.log('setting up pipelines...')
        const romance_to_en_translator = await pipeline('translation', 'xenova/opus-mt-en-ROMANCE');
        const en_to_romance_translator = await pipeline('translation', 'xenova/opus-mt-en-ROMANCE');

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





  const translate = async (text, source_lang, target_lang) => {
    console.log( 'translating ', text, source_lang, target_lang);
    if (!models_loaded) {
      console.log('Translator not ready');
      return;
    }

    let translator;

    if (source_lang === 'en') {
      translator = enRomanceTranslator;
      text = `>>${target_lang}<< ${text}`
    } else {
      translator = romanceEnTranslator;
    }

    const translated_text = await translator(text);
    console.log(translated_text);
    return translated_text;
  }


  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Translate</Text>

      <Button
        title="Translate"
        onPress={() => {
          translate("test me", "en", "pt");
        }}
      />
      
      <View style={styles.translationContainer}>
        {/* Source Language Section */}
        <View style={styles.languageSection}>
          <View style={styles.languageHeader}>
            <TouchableOpacity style={styles.languageSelector}>
              <Text style={styles.languageText}>{sourceLang.toUpperCase()}</Text>
            </TouchableOpacity>
            <TouchableOpacity>
              <Ionicons name="mic" size={24} color="black" />
            </TouchableOpacity>
          </View>
          <TextInput
            style={styles.textInput}
            multiline
            placeholder="Enter text"
            value={sourceText}
            onChangeText={setSourceText}
          />
        </View>

        {/* Language Swap Button */}
        <TouchableOpacity 
          style={styles.swapButton}
          onPress={() => {
            const tempLang = sourceLang;
            setSourceLang(targetLang);
            setTargetLang(tempLang);
            const tempText = sourceText;
            setSourceText(targetText);
            setTargetText(tempText);
          }}
        >
          <Ionicons name="swap-vertical" size={24} color="white" />
        </TouchableOpacity>

        {/* Target Language Section */}
        <View style={styles.languageSection}>
          <View style={styles.languageHeader}>
            <TouchableOpacity style={styles.languageSelector}>
              <Text style={styles.languageText}>{targetLang.toUpperCase()}</Text>
            </TouchableOpacity>
            <TouchableOpacity>
              <Ionicons name="mic" size={24} color="black" />
            </TouchableOpacity>
          </View>
          <TextInput
            style={styles.textInput}
            multiline
            placeholder="Translation"
            value={targetText}
            editable={false}
          />
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    padding: 20,
    textAlign: 'center',
  },
  translationContainer: {
    flex: 1,
    marginHorizontal: 16,
  },
  languageSection: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 16,
    marginVertical: 8,
  },
  languageHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  languageSelector: {
    backgroundColor: '#e0e0e0',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  languageText: {
    fontSize: 16,
    fontWeight: '600',
  },
  textInput: {
    flex: 1,
    fontSize: 18,
    textAlignVertical: 'top',
  },
  swapButton: {
    position: 'absolute',
    right: 16,
    top: '50%',
    marginTop: -20,
    backgroundColor: '#007AFF',
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
});

