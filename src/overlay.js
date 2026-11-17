// 분석 결과(result.json)를 원본 영상 위에 그리기 위한 계산·그리기 함수 모음.
// 화면 구성 요소(React)와 떼어 두어 단위 테스트(tests/overlay.test.js)로 확인할 수 있게 했다.

// 재생 시각(초)을 결과 배열의 프레임 번호로 바꾼다.
// 반올림해서 가장 가까운 프레임을 고르고, 영상 끝을 넘으면 마지막 프레임으로 고정한다
// (소리 트랙이 영상보다 길면 브라우저가 알려 주는 길이가 프레임 수보다 길 수 있다).
export function frameAt(seconds, video) {
  const index = Math.round(seconds * video.fps);
  return Math.min(Math.max(index, 0), video.frames - 1);
}

// 이 프레임이 분석 구간(스키어가 80px 이상으로 충분히 크게 보이는 구간) 안인지.
export function inSegment(result, index) {
  return result.frames.seg[index] === 1;
}

// 무릎 굽힘 값을 화면 글자로. 값이 없으면(가려짐·분석 구간 밖) '보류'로 보여 준다.
export function kneeText(value) {
  return value === null || value === undefined ? '보류' : Math.round(value) + '°';
}

// 타임라인에 칠할 분석 구간들을 영상 전체 길이 대비 백분율(left, width)로 바꾼다.
export function timelineParts(result) {
  const total = result.video.frames / result.video.fps;
  return result.segments.map(s => ({ left: 100 * s.start / total, width: 100 * (s.end - s.start) / total }));
}

// result.frames.body 안에서 각 관절이 몇 번째에 있는지 (MoveNet/COCO 순서 17점).
// 왼쪽·오른쪽은 화면 기준이 아니라 스키어 자신의 왼쪽·오른쪽이다.
export const JOINT = {
  nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8, leftWrist: 9, rightWrist: 10,
  leftHip: 11, rightHip: 12, leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16
};

// 스키어 위에 그릴 선 목록. 두 관절(from → to)을 잇는다. side는 색을 정한다.
// 선을 더하거나 빼려면 이 표만 고치면 된다 (예: 2단계에서 스키·폴 선 추가).
export const BODY_LINES = [
  { side: 'centre', from: 'leftShoulder', to: 'rightShoulder' },   // 어깨선
  { side: 'centre', from: 'leftHip', to: 'rightHip' },             // 허리선(양쪽 골반)
  { side: 'left', from: 'leftShoulder', to: 'leftHip' },           // 몸통 왼쪽
  { side: 'left', from: 'leftShoulder', to: 'leftElbow' },         // 왼쪽 위팔
  { side: 'left', from: 'leftElbow', to: 'leftWrist' },            // 왼쪽 아래팔
  { side: 'left', from: 'leftHip', to: 'leftKnee' },               // 왼쪽 허벅지
  { side: 'left', from: 'leftKnee', to: 'leftAnkle' },             // 왼쪽 정강이
  { side: 'right', from: 'rightShoulder', to: 'rightHip' },        // 몸통 오른쪽
  { side: 'right', from: 'rightShoulder', to: 'rightElbow' },      // 오른쪽 위팔
  { side: 'right', from: 'rightElbow', to: 'rightWrist' },         // 오른쪽 아래팔
  { side: 'right', from: 'rightHip', to: 'rightKnee' },            // 오른쪽 허벅지
  { side: 'right', from: 'rightKnee', to: 'rightAnkle' }           // 오른쪽 정강이
];

// result.frames.skis 안에서 각 점이 몇 번째에 있는지 (24점 모델의 16~23번을 그대로 옮긴 8점).
// 몸 17점(MoveNet)과 스키 8점(24점 YOLO 모델)은 서로 다른 모델이 찾은 점이라, 발목과 스키를 선으로 잇지 않는다.
export const SKI_POINT = {
  rightSkiTip: 0, rightToes: 1, rightHeel: 2, rightSkiTail: 3,
  leftSkiTip: 4, leftToes: 5, leftHeel: 6, leftSkiTail: 7
};

// 스키·발 선: 스키 한 짝(앞끝 → 뒤끝)과 부츠 바닥(발끝 → 뒤꿈치). 폴은 모델이 거의 못 맞혀 그리지 않는다.
export const SKI_LINES = [
  { side: 'left', from: 'leftSkiTip', to: 'leftSkiTail' },         // 왼쪽 스키
  { side: 'left', from: 'leftToes', to: 'leftHeel' },              // 왼쪽 부츠 바닥
  { side: 'right', from: 'rightSkiTip', to: 'rightSkiTail' },      // 오른쪽 스키
  { side: 'right', from: 'rightToes', to: 'rightHeel' }            // 오른쪽 부츠 바닥
];

// 선 색. 왼쪽 빨강·오른쪽 노랑은 오른편 무릎 수치 글자색과 맞췄다.
// 가운데(어깨선·허리선·머리)는 흰 눈 위에서도 보이도록 흰색 대신 하늘색으로 했다.
// 화면 범례의 색(src/style.css의 .legend)도 같이 바꿔야 한다.
export const COLOURS = { left: '#ff5a4f', right: '#ffc233', centre: '#33c8ff' };

// 한 프레임에 그릴 선들을 계산한다.
//   points: 17점 [x, y] (화면 대비 비율 0~1), seen: 17점 각각 모델이 확실히 찾았으면 1
//   돌려주는 값: [{ side, from: [x, y], to: [x, y], sure }]
// 선의 양 끝 관절을 둘 다 확실히 찾았을 때만 sure = true (아니면 흐리게 그린다).
export function bodyLines(points, seen) {
  return linesFrom(BODY_LINES, JOINT, points, seen);
}

// 한 프레임의 스키·발 선. points·seen은 result.frames.skis / skis_seen의 한 프레임(8점).
// 스키 점이 없는 결과(섞어 쓰기 이전 파일)나 24점 모델이 스키어를 놓친 프레임이면 빈 목록.
export function skiLines(points, seen) {
  return points && seen ? linesFrom(SKI_LINES, SKI_POINT, points, seen) : [];
}

// 선 표(lines)와 점 번호 표(index)로 그릴 선 좌표를 만든다. bodyLines·skiLines가 함께 쓴다.
function linesFrom(lines, index, points, seen) {
  return lines.map(line => {
    const a = index[line.from], b = index[line.to];
    return { side: line.side, from: points[a], to: points[b], sure: seen[a] === 1 && seen[b] === 1 };
  });
}

// 캔버스에 한 프레임을 그린다. width·height는 캔버스 크기(= 화면에 보이는 영상 크기)다.
// 좌표가 비율로 저장돼 있어서 '비율 × 캔버스 크기'로 어떤 화면 크기에서도 같은 자리에 그려진다.
export function drawFrame(context, result, index, width, height) {
  context.clearRect(0, 0, width, height);               // 지난 프레임 그림 지우기
  const points = result.frames.body[index];
  if (!points) return;                                   // 아직 스키어를 찾기 전인 프레임: 아무것도 안 그림
  const active = inSegment(result, index);
  const thickness = Math.max(2, width / 320);            // 화면이 커지면 선도 굵게 (최소 2px)

  // 1) 분석 구간 안이면 스키어 상자(흰 테두리)
  const box = result.frames.box[index];
  if (box && active) {
    context.strokeStyle = 'rgba(255,255,255,0.9)';
    context.lineWidth = thickness / 2;
    context.strokeRect(box[0] * width, box[1] * height, (box[2] - box[0]) * width, (box[3] - box[1]) * height);
  }

  // 2) 관절선과 스키·발 선. 분석 구간 밖이거나 확실하지 않은 선은 투명도 35%로 흐리게
  context.lineWidth = thickness;
  context.lineCap = 'round';
  const skis = result.frames.skis ? skiLines(result.frames.skis[index], result.frames.skis_seen[index]) : [];
  for (const line of [...bodyLines(points, result.frames.seen[index]), ...skis]) {
    context.globalAlpha = active && line.sure ? 1 : 0.35;
    context.strokeStyle = COLOURS[line.side];
    context.beginPath();
    context.moveTo(line.from[0] * width, line.from[1] * height);
    context.lineTo(line.to[0] * width, line.to[1] * height);
    context.stroke();
  }

  // 3) 머리: 코 위치에 점 하나 (MoveNet에는 '머리' 점이 따로 없어서 코로 대신한다)
  const nose = points[JOINT.nose];
  context.globalAlpha = active && result.frames.seen[index][JOINT.nose] === 1 ? 1 : 0.35;
  context.fillStyle = COLOURS.centre;
  context.beginPath();
  context.arc(nose[0] * width, nose[1] * height, thickness * 1.5, 0, 2 * Math.PI);
  context.fill();
  context.globalAlpha = 1;                               // 다음 그리기에 투명도가 남지 않게 되돌린다
}
