// 앱의 시작점. 주소에 맞는 페이지를 골라 머리글 메뉴와 본문을 그린다.
// 페이지 이동은 일반 링크(<a href>)로 한다: 새로고침·뒤로 가기·주소 직접 입력이 브라우저 기본 동작 그대로 된다.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { pages, findPage } from './routes.js';
import { Analysis } from './analysis.jsx';
import './style.css';

// 지금 주소의 페이지 (없는 주소면 undefined)
const active = findPage(window.location.pathname);
const title = active?.title ?? '페이지를 찾을 수 없습니다';
// 브라우저 탭 제목: 홈은 서비스 이름만, 다른 페이지는 '페이지 이름 · 서비스 이름'
document.title = active?.path === '/' ? title : title + ' · 스키 자세 코칭';

function App() {
  return <>
    {/* 메뉴: 지금 페이지는 aria-current로 표시해 화면 낭독기와 강조 스타일이 함께 쓴다 */}
    <header><nav aria-label="주 메뉴">{pages.map(page =>
      <a key={page.path} href={page.path} aria-current={page === active ? 'page' : undefined}>{page.label}</a>
    )}</nav></header>
    <main>
      <h1>{title}</h1>
      {/* 없는 주소면 설명 대신 홈으로 돌아가는 링크 */}
      {active ? <p>{active.description}</p> : <a href="/">홈으로 돌아가기</a>}
      {active?.path === '/analysis' && <Analysis />}
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(<App />);
