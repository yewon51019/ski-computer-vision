# 스키 주행 자세 코칭

현재 웹 앱의 범위는 홈(/), 주행 분석(/analysis), 분석 결과(/results)의 기본 탐색이다.

## 로컬 실행

프로젝트 루트에서 Node 24와 npm으로 다음을 실행한다.

```sh
npm install
npm run dev
```

개발 서버가 출력하는 주소로 접속한다. 검증 명령은 npm test, npm run lint, npm run build다. 기본 경로 검사는 외부 패키지 없이 실행할 수 있다.

## 문서와 데이터

개발 순서는 [TASKS.md](TASKS.md), 구현 절차는 [AGENTS.md](AGENTS.md)에 있다. docs/에는 제품 요구사항, MVP, 기술설계 및 테스트 결과가 있다. 기존 Python 실험에서 사용한 EPFL 원본 데이터와 분석 CSV는 웹 앱 실행에 필요하지 않다.
