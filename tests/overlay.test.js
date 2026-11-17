// src/overlay.js의 계산 함수 시험 (그리기 자체는 브라우저에서 확인). 실행: npm test
import test from 'node:test';
import assert from 'node:assert/strict';
import { bodyLines, frameAt, inSegment, JOINT, kneeText, SKI_POINT, skiLines, timelineParts } from '../src/overlay.js';

// 시험용 가짜 결과: 24fps, 48프레임(2초). 12~23번 프레임(0.5~1초)만 분석 구간
const result = {
  video: { fps: 24, frames: 48 },
  segments: [{ start: 0.5, end: 1 }],
  frames: { seg: Array.from({ length: 48 }, (_, i) => (i >= 12 && i < 24 ? 1 : 0)) }
};

test('재생 시각을 프레임 번호로 바꾸고 범위를 벗어나지 않는다', () => {
  assert.equal(frameAt(0, result.video), 0);
  assert.equal(frameAt(0.5, result.video), 12);
  assert.equal(frameAt(99, result.video), 47);
  assert.equal(frameAt(-1, result.video), 0);
});

test('분석 구간 안팎을 구분한다', () => {
  assert.equal(inSegment(result, 11), false);
  assert.equal(inSegment(result, 12), true);
  assert.equal(inSegment(result, 24), false);
});

test('굽힘량이 없으면 보류로 표시한다', () => {
  assert.equal(kneeText(null), '보류');
  assert.equal(kneeText(18.6), '19°');
  assert.equal(kneeText(0), '0°');
});

test('타임라인 구간을 영상 길이 대비 비율로 계산한다', () => {
  assert.deepEqual(timelineParts(result), [{ left: 25, width: 25 }]);
});

test('몸 17점에서 머리를 뺀 전신 선 12개를 만들고, 양 끝을 모두 찾았을 때만 확실하다고 본다', () => {
  const points = Array.from({ length: 17 }, (_, i) => [i / 100, i / 100]);
  const seen = Array(17).fill(1);
  seen[JOINT.leftWrist] = 0;                                   // the left hand was not found
  const lines = bodyLines(points, seen);
  assert.equal(lines.length, 12);
  const forearm = lines.find(l => l.side === 'left' && l.to === points[JOINT.leftWrist]);
  assert.equal(forearm.sure, false);
  assert.deepEqual(forearm.from, points[JOINT.leftElbow]);
  assert.equal(lines.filter(l => !l.sure).length, 1);
  assert.deepEqual(lines.filter(l => l.side === 'centre').map(l => [l.from, l.to]),
    [[points[JOINT.leftShoulder], points[JOINT.rightShoulder]], [points[JOINT.leftHip], points[JOINT.rightHip]]]);
});

test('스키·발 8점에서 스키 2개와 부츠 바닥 2개 선을 만들고, 스키 점이 없으면 아무것도 그리지 않는다', () => {
  const points = Array.from({ length: 8 }, (_, i) => [i / 10, i / 10]);
  const seen = Array(8).fill(1);
  seen[SKI_POINT.rightSkiTail] = 0;                            // 오른쪽 스키 뒤끝을 못 찾음
  const lines = skiLines(points, seen);
  assert.equal(lines.length, 4);
  const rightSki = lines.find(l => l.side === 'right' && l.from === points[SKI_POINT.rightSkiTip]);
  assert.deepEqual(rightSki.to, points[SKI_POINT.rightSkiTail]);
  assert.equal(rightSki.sure, false);
  assert.equal(lines.filter(l => l.sure).length, 3);
  assert.deepEqual(skiLines(null, null), []);                  // 24점 모델이 스키어를 놓친 프레임
  assert.deepEqual(skiLines(undefined, undefined), []);        // 섞어 쓰기 이전 결과 파일
});
