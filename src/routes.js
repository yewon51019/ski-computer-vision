// 앱의 페이지 목록. 메뉴 순서도 이 순서를 따른다.
//   path: 주소, label: 메뉴 글자, title: 페이지 제목(h1), description: 제목 아래 설명
export const pages = [
  { path: '/', label: '홈', title: '스키 자세 코칭', description: '주행 분석과 결과 화면으로 이동하세요.' },
  { path: '/analysis', label: '주행 분석', title: '주행 분석', description: '영상을 올리고 스키어를 고르면 전신 관절선을 영상 위에 그리고 무릎 굽힘을 표시합니다.' },
  { path: '/results', label: '분석 결과', title: '분석 결과', description: '표시할 분석 결과가 없습니다.' }
];

// 주소(pathname)에 맞는 페이지를 찾는다. 끝의 '/'는 무시한다(/analysis/ = /analysis).
// 맞는 페이지가 없으면 undefined를 돌려준다.
export function findPage(pathname) {
  const path = pathname.replace(/\/+$/, '') || '/';      // 끝 '/'를 모두 지우고, 다 지워지면 홈('/')
  return pages.find(page => page.path === path);
}
