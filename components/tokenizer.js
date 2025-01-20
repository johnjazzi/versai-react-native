class SentencePieceTokenizer {
    constructor(vocab, mergeRules) {
      this.vocab = new Map();
      this.reverseVocab = new Map();
      this.mergeRules = new Map();
      
      // Special tokens
      this.specialTokens = {
        pad: '<pad>',
        eos: '</s>',
        unk: '<unk>',
        bos: '<s>'
      };
      
      // Add special tokens first
      this.addSpecialTokens();
      
      // Load vocabulary and merge rules
      this.loadVocab(vocab);
      this.loadMergeRules(mergeRules);
    }
  
    addSpecialTokens() {
      Object.values(this.specialTokens).forEach((token, index) => {
        this.vocab.set(token, index);
        this.reverseVocab.set(index, token);
      });
    }
  
    loadVocab(vocab) {
      vocab.forEach((word, index) => {
        const id = this.vocab.size;
        this.vocab.set(word, id);
        this.reverseVocab.set(id, word);
      });
    }
  
    loadMergeRules(rules) {
      rules.forEach(([pair, merged]) => {
        this.mergeRules.set(pair, merged);
      });
    }
  
    // SentencePiece-style tokenization
    tokenize(text) {
      // Normalize text and add spaces between characters
      const normalized = this.normalizeText(text);
      
      // Start with character-level tokens
      let tokens = this.characterTokenize(normalized);
      
      // Apply merge rules iteratively
      tokens = this.applyMergeRules(tokens);
      
      // Convert to token ids
      const tokenIds = [this.vocab.get(this.specialTokens.bos)];
      for (const token of tokens) {
        const id = this.vocab.get(token) || this.vocab.get(this.specialTokens.unk);
        tokenIds.push(id);
      }
      tokenIds.push(this.vocab.get(this.specialTokens.eos));
      
      return tokenIds;
    }
  
    normalizeText(text) {
      // Basic normalization: lowercase and handle whitespace
      return text.toLowerCase().trim().replace(/\s+/g, ' ');
    }
  
    characterTokenize(text) {
      // Add space to beginning of each word
      const withSpaces = text.replace(/\S/g, ' $&');
      return withSpaces.split('');
    }
  
    applyMergeRules(tokens) {
      let changed = true;
      while (changed) {
        changed = false;
        
        for (let i = 0; i < tokens.length - 1; i++) {
          const pair = tokens[i] + tokens[i + 1];
          const merged = this.mergeRules.get(pair);
          
          if (merged) {
            tokens.splice(i, 2, merged);
            changed = true;
            break;
          }
        }
      }
      return tokens;
    }
  
    decode(tokenIds) {
      return tokenIds
        .map(id => this.reverseVocab.get(id))
        .filter(token => !Object.values(this.specialTokens).includes(token))
        .join('')
        .trim()
        .replace(/▁/g, ' '); // Replace SentencePiece space marker
    }
  
    // Utility method to pad sequences to same length
    padSequence(tokenIds, maxLength) {
      const padId = this.vocab.get(this.specialTokens.pad);
      const padding = Array(Math.max(0, maxLength - tokenIds.length)).fill(padId);
      return tokenIds.concat(padding).slice(0, maxLength);
    }
  }