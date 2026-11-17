// src/routes.js의 주소 → 페이지 찾기 시험. 실행: npm test
import test from 'node:test';
import assert from 'node:assert/strict';
import { findPage } from '../src/routes.js';

test('기본 경로와 끝 슬래시', () => {
  // 세 페이지 모두, 주소 끝에 '/'가 붙어도 같은 페이지로 찾아야 한다 (예: /analysis/)
  for (const path of ['/', '/analysis', '/results']) {
    assert.equal(findPage(path)?.path, path);
    assert.equal(findPage(path + '/')?.path, path);
  }
});

test('등록되지 않은 주소', () => {
  // 없는 주소는 undefined → 화면은 '페이지를 찾을 수 없습니다'를 보여 준다
  assert.equal(findPage('/unknown'), undefined);
});
