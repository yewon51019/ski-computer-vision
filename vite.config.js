import { defineConfig } from 'vite';

// 개발 서버(npm run dev, 5173번) 설정.
// 분석 API(FastAPI)는 8000번에서 돈다. 화면이 부르는 /api/... 요청을 그쪽으로 넘겨 주어
// 브라우저 입장에서는 같은 주소에서 오는 것처럼 보이게 한다(다른 주소 요청 제한을 피함).
export default defineConfig({
  server: { proxy: { '/api': 'http://localhost:8000' } }
});
