module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['module:@react-native/babel-preset'],
    plugins: [
      '@babel/plugin-syntax-import-assertions',
      '@babel/plugin-proposal-export-namespace-from',
      'babel-plugin-transform-import-meta'
    ]
  };
};