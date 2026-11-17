// 코드 검사(npm run lint) 설정. src/ 아래 JS·JSX만 검사한다.
export default [{
  files: ['src/**/*.{js,jsx}'],
  languageOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    parserOptions: { ecmaFeatures: { jsx: true } },
    // 브라우저가 기본으로 주는 이름들. 여기 적어야 '정의되지 않은 이름' 오류로 잡히지 않는다
    globals: { window: 'readonly', document: 'readonly', fetch: 'readonly', FormData: 'readonly', URL: 'readonly',
      Blob: 'readonly', setInterval: 'readonly', clearInterval: 'readonly', requestAnimationFrame: 'readonly',
      cancelAnimationFrame: 'readonly' }
  },
  // 실제 오류로 이어지는 것만 막는다: 정의 안 된 이름, 실행될 수 없는 코드, 같은 키 두 번
  rules: { 'no-undef': 'error', 'no-unreachable': 'error', 'no-dupe-keys': 'error' }
}];
