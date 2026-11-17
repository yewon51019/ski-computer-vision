// 주행 분석 페이지(/analysis)의 화면.
// 흐름: 영상 고르기 → 서버에 올리기 → 서버가 사람 찾기 → 사용자가 스키어 고르기 → 서버가 관절 분석 → 결과 재생
// 영상 파일은 브라우저에서 바로 재생하고(서버에서 다시 받지 않음), 서버에서는 좌표 결과(result.json)만 받아
// 영상 위 캔버스에 관절선을 그린다.
import React, { useEffect, useRef, useState } from 'react';
import { drawFrame, frameAt, kneeText, timelineParts } from './overlay.js';

// 서버 API 호출 도우미. 주소 앞에 /api/jobs를 붙이고, 응답 JSON을 돌려준다.
// 실패하면 서버가 보낸 이유(detail)나 기본 문구로 오류를 던진다 → 화면 위 빨간 안내로 보인다.
const api = (path, options) => fetch('/api/jobs' + path, options).then(async response => {
  const body = await response.json().catch(() => ({}));       // 응답이 JSON이 아니어도 멈추지 않게
  if (!response.ok) throw new Error(body.detail || '서버에 연결하지 못했습니다.');
  return body;
});

// 기다리는 동안 보여 줄 단계 이름
const STAGE = { upload: '영상을 올리는 중', people: '사람을 찾는 중', pose: '관절을 분석하는 중' };

// 진행 문구. 관절 분석은 두 모델을 차례로 돌린다(섞어 쓰기): 먼저 24점 모델이 스키를 찾고(part: 'skis'),
// 이어서 MoveNet이 몸 관절과 무릎을 잰다. 프레임 수 막대가 두 번 차므로 몇 번째 부분인지 함께 보여 준다.
function stageText(step, progress) {
  if (step !== 'pose' || !progress || progress.stage !== 'pose') return STAGE[step];
  return progress.part === 'skis' ? '스키를 찾는 중 (1/2)' : '관절을 분석하는 중 (2/2)';
}

export function Analysis() {
  // 화면 단계: pick(영상 고르기) → upload(올리는 중) → people(사람 찾는 중) → choose(스키어 고르기)
  //            → pose(관절 분석 중) → done(결과)
  const [step, setStep] = useState('pick');
  const [error, setError] = useState('');            // 화면 위에 보여 줄 오류 문구
  const [job, setJob] = useState(null);              // 서버 작업 정보 { id, candidates(후보들), video }
  const [progress, setProgress] = useState(null);    // 서버가 알려 주는 진행 { stage, done, total }
  const [source, setSource] = useState('');          // 고른 영상 파일을 브라우저에서 재생하기 위한 임시 주소
  const [result, setResult] = useState(null);        // 서버가 만든 분석 결과(result.json)

  // 영상이 바뀌거나 화면을 떠날 때 임시 주소를 풀어 메모리를 돌려준다
  useEffect(() => () => { if (source) URL.revokeObjectURL(source); }, [source]);

  // 서버가 일하는 동안(사람 찾기·관절 분석) 1초마다 상태를 물어본다
  useEffect(() => {
    if (!job || (step !== 'people' && step !== 'pose')) return undefined;
    let stopped = false;                              // 이 반복이 정리된 뒤 늦게 온 응답은 무시하기 위한 표시
    const timer = setInterval(async () => {
      try {
        const state = await api('/' + job.id);
        if (stopped) return;
        setProgress(state.progress);
        if (state.state === 'error') throw new Error(state.error);
        if (state.state === 'choose') { setJob({ ...job, ...state }); setStep('choose'); }   // 후보 목록 도착
        if (state.state === 'done') { setResult(await api('/' + job.id + '/result')); setStep('done'); }
      } catch (problem) {
        if (!stopped) { setError(problem.message); setStep('pick'); }   // 문제가 생기면 처음 화면으로
      }
    }, 1000);
    return () => { stopped = true; clearInterval(timer); };
  }, [job, step]);

  // 영상 파일을 골랐을 때: 이전 작업을 정리하고, 화면 재생 준비 후 서버로 올린다
  async function upload(file) {
    if (!file) return;
    reset();
    setSource(URL.createObjectURL(file));
    setStep('upload');
    try {
      const form = new FormData();
      form.append('file', file);
      const created = await api('', { method: 'POST', body: form });
      setJob({ id: created.id });
      setStep('people');                              // 서버가 바로 사람 찾기를 시작한다
    } catch (problem) {
      setError(problem.message);
      setStep('pick');
    }
  }

  // 후보 중 한 명을 골랐을 때: 서버에 그 사람(track 번호)의 관절 분석을 요청한다
  async function choose(track) {
    setProgress(null);
    setResult(null);
    try {
      await api('/' + job.id + '/select', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track }) });
      setStep('pose');
    } catch (problem) {
      setError(problem.message);
    }
  }

  // 처음 상태로 되돌린다. 서버에 올린 영상·결과는 임시 사본이므로 지워 달라고 요청한다(응답은 기다리지 않음)
  function reset() {
    if (job) api('/' + job.id, { method: 'DELETE' }).catch(() => {});
    setJob(null); setResult(null); setProgress(null); setError(''); setSource(''); setStep('pick');
  }

  // 단계에 따라 아래 중 하나가 보인다: 올리기 칸 / 진행 막대 / 후보 목록 / 결과
  return <section className="analysis">
    {error && <p className="error" role="alert">{error}</p>}

    {step === 'pick' && <label className="drop">
      <strong>주행 영상을 올려 주세요</strong>
      <span>MP4 또는 MOV. 스키어가 화면에서 80픽셀 이상으로 보이는 구간만 분석됩니다.</span>
      <input type="file" accept="video/mp4,video/quicktime,.mp4,.mov" onChange={event => upload(event.target.files[0])} />
    </label>}

    {(step === 'upload' || step === 'people' || step === 'pose') && <div className="working" aria-live="polite">
      <p>{stageText(step, progress)}{progress && progress.stage === step ? ` · ${progress.done} / ${progress.total} 프레임` : ''}</p>
      <progress max={progress && progress.stage === step ? progress.total : undefined}
        value={progress && progress.stage === step ? progress.done : undefined} />
    </div>}

    {step === 'choose' && <div>
      <h2>분석할 스키어를 골라 주세요</h2>
      {job.candidates.length === 0
        ? <p>분석할 수 있을 만큼 크게 찍힌 사람이 없습니다. 더 가까이에서 찍은 영상을 올려 주세요.</p>
        : <ul className="people">{job.candidates.map(person => <li key={person.track}>
          <button type="button" onClick={() => choose(person.track)}>
            <img src={`/api/jobs/${job.id}/thumb/${person.track}`} alt="" width="160" height="160" />
            <span>{person.best_seconds.toFixed(1)}초 부근</span>
            <small>분석 가능 {person.analysable_seconds.toFixed(1)}초</small>
          </button>
        </li>)}</ul>}
      <button type="button" onClick={reset}>다른 영상 올리기</button>
    </div>}

    {step === 'done' && result && <Result source={source} result={result}
      onPickAgain={job.candidates.length > 1 ? () => setStep('choose') : null} onReset={reset} />}
  </section>;
}

// 결과 화면: 원본 영상 + 그 위 관절선 캔버스 + 타임라인 + 오른쪽 무릎 수치와 버튼들.
//   onPickAgain: 후보가 두 명 이상일 때만 주어진다('다른 사람 선택' 버튼). onReset: '새 영상' 버튼.
function Result({ source, result, onPickAgain, onReset }) {
  const video = useRef(null);
  const canvas = useRef(null);
  const [index, setIndex] = useState(0);             // 지금 화면에 보이는 프레임 번호 (수치·타임라인 표시용)
  const total = result.video.frames / result.video.fps;   // 영상 길이(초), 프레임 수 기준

  useEffect(() => {
    const element = video.current;
    let handle;                                       // 예약해 둔 다음 그리기의 번호 (정리할 때 취소용)
    // 주어진 재생 시각의 프레임을 캔버스에 그리고, 오른쪽 수치도 그 프레임으로 맞춘다.
    // 캔버스 크기는 그릴 때마다 화면에 보이는 영상 크기에 맞춘다 (창 크기·화면 회전 대응).
    const draw = seconds => {
      const frame = frameAt(seconds, result.video);
      const target = canvas.current;
      if (target.width !== element.clientWidth || target.height !== element.clientHeight) {
        target.width = element.clientWidth;
        target.height = element.clientHeight;
      }
      drawFrame(target.getContext('2d'), result, frame, target.width, target.height);
      setIndex(frame);
    };
    // 그리기 반복은 결과 화면이 떠 있는 동안 하나만 둔다.
    // requestVideoFrameCallback: 브라우저가 새 영상 프레임을 화면에 낼 때마다 불러 주고,
    // meta.mediaTime에 그 프레임의 정확한 재생 시각을 준다. 지원하지 않는 브라우저는 화면 갱신마다 그린다.
    const paint = (now, meta) => {
      draw(meta ? meta.mediaTime : element.currentTime);
      handle = element.requestVideoFrameCallback
        ? element.requestVideoFrameCallback(paint) : requestAnimationFrame(() => paint());
    };
    paint();
    // 일시정지 중 이동(타임라인 클릭 등)하면 새 프레임 알림이 오지 않으므로 직접 한 번 그린다.
    // 여기서 paint를 부르면 반복이 하나 더 생겨 쌓이므로 draw만 부른다 (2026-11-17 고친 결함).
    const seeked = () => draw(element.currentTime);
    element.addEventListener('seeked', seeked);
    // 결과 화면을 떠날 때: 이벤트와 예약된 그리기를 정리한다
    return () => {
      element.removeEventListener('seeked', seeked);
      if (element.cancelVideoFrameCallback) element.cancelVideoFrameCallback(handle); else cancelAnimationFrame(handle);
    };
  }, [result]);

  // 타임라인을 누른 위치의 비율(0~1)만큼 영상 길이를 곱해 그 시각으로 이동한다
  function seek(event) {
    const bar = event.currentTarget.getBoundingClientRect();
    video.current.currentTime = total * (event.clientX - bar.left) / bar.width;
  }

  // 결과를 JSON 파일로 내려받는다 (기기에 보관하기로 한 결정에 따라 서버에는 남기지 않음)
  function save() {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(result)], { type: 'application/json' }));
    link.download = 'ski-analysis.json';
    link.click();
    URL.revokeObjectURL(link.href);
  }

  const { left, right } = result.frames;              // 프레임별 왼쪽·오른쪽 무릎 굽힘(도), 보류면 null
  return <div className="result">
    <div className="player">
      <div className="stage">
        <video ref={video} src={source} controls playsInline muted />
        <canvas ref={canvas} aria-hidden="true" />
      </div>
      <div className="timeline" onClick={seek} role="slider" aria-label="분석 구간 타임라인"
        aria-valuemin={0} aria-valuemax={Math.round(total)} aria-valuenow={Math.round(index / result.video.fps)} tabIndex={0}>
        {timelineParts(result).map((part, i) => <i key={i} style={{ left: part.left + '%', width: part.width + '%' }} />)}
        <b style={{ left: 100 * index / result.video.frames + '%' }} />
      </div>
      <p className="legend"><span className="on" /> 분석 구간 <span className="off" /> 보류 (너무 멀거나 화면 밖)</p>
      <p className="legend"><span className="leftSide" /> 스키어의 왼쪽 <span className="rightSide" /> 오른쪽
        <span className="centre" /> 어깨선·허리선·머리 · 흐린 선은 확실하지 않은 부분</p>
      {result.frames.skis && <p className="legend">아래쪽 긴 선은 스키, 짧은 선은 부츠 바닥(발끝–뒤꿈치)입니다. 몸 선과 다른 모델로 찾아 서로 잇지 않았습니다.</p>}
    </div>
    <div className="numbers">
      <p><span>왼쪽 무릎 굽힘</span><strong className="leftKnee">{kneeText(left[index])}</strong></p>
      <p><span>오른쪽 무릎 굽힘</span><strong className="rightKnee">{kneeText(right[index])}</strong></p>
      <small>화면상 추정값입니다. 전체 {result.summary.seconds.toFixed(1)}초 중 {result.summary.analysed_seconds.toFixed(1)}초를 분석했습니다.</small>
      <button type="button" onClick={save}>결과 파일 저장</button>
      {onPickAgain && <button type="button" onClick={onPickAgain}>다른 사람 선택</button>}
      <button type="button" onClick={onReset}>새 영상</button>
    </div>
  </div>;
}
