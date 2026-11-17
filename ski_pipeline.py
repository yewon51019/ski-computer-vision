"""분석 두 단계를 각자의 Python 환경에서 실행한다. 웹 서버(server/app.py)가 이 모듈을 부른다.

왜 따로 실행하나
  사람 검출기(PyTorch, .yolo-venv)와 MoveNet(TensorFlow, .movenet-gpu)이 서로 다른 환경에 있어서
  한 프로세스에 함께 올릴 수 없다. 그래서 단계마다 해당 환경의 python.exe로 스크립트를 따로 돌린다.
  이 모듈 자체는 표준 라이브러리만 쓰므로 웹 서버 환경(.web-venv)에서 불러올 수 있다.

  find_people(영상, 작업 폴더)          → 후보 dict (사용자가 고를 수 있는 사람들)
  analyse(영상, 작업 폴더, 추적 번호)    → 결과 dict (브라우저가 그리는 것)
  read_progress(작업 폴더)               → {"stage", "done", "total"} 또는 None

명령줄에서 직접 (프로젝트 루트에서)
  python ski_pipeline.py people clip.mp4 data/jobs/clip
  python ski_pipeline.py analyse clip.mp4 data/jobs/clip 3
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLO_PYTHON = ROOT / '.yolo-venv' / 'Scripts' / 'python.exe'     # 1단계: 사람 찾기
MOVENET_PYTHON = ROOT / '.movenet-gpu' / 'python.exe'            # 2단계: 관절 분석


def _run(python, script, *arguments):
    """python으로 script를 실행하고 표준 출력을 돌려준다. 실패하면 오류 출력 끝 2000자를 담아 예외를 던진다."""
    done = subprocess.run([str(python), str(ROOT / script), *map(str, arguments)], cwd=ROOT,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    if done.returncode:
        raise RuntimeError('%s failed (exit %d): %s' % (script, done.returncode, done.stderr[-2000:]))
    return done.stdout


def find_people(video, job_dir):
    """1단계: 영상의 모든 사람을 찾아 이어 붙이고(person_tracks.py), 만들어진 후보 목록을 돌려준다."""
    job_dir = Path(job_dir)
    _run(YOLO_PYTHON, 'person_tracks.py', Path(video).resolve(), '--out-dir', job_dir.resolve(), '--progress')
    return json.loads((job_dir / 'candidates.json').read_text(encoding='utf-8'))


def analyse(video, job_dir, track):
    """2단계: 고른 한 명(track)의 관절을 분석하고, 결과를 돌려준다.

    섞어 쓰기(2026-11-18 사용자 결정): 몸·무릎은 MoveNet, 스키·발은 24점 YOLO 모델.
      ① ski24_analyze.py (.yolo-venv)    → ski24.json (24점, 이 중 스키·발 8점만 쓴다)
      ② ski_analyze.py   (.movenet-gpu)  → 몸 17점·무릎을 구하고 ski24.json의 스키·발 점을 붙여 result.json
    result.json은 ②가 끝날 때 한 번만 생기므로, 스키 선이 빠진 결과가 화면에 먼저 뜨는 일이 없다.
    """
    job_dir = Path(job_dir)
    common = (Path(video).resolve(), '--job-dir', job_dir.resolve(), '--track', track, '--progress')
    _run(YOLO_PYTHON, 'ski24_analyze.py', *common, '--out', 'ski24.json')
    _run(MOVENET_PYTHON, 'ski_analyze.py', *common, '--skis', 'ski24.json')
    return json.loads((job_dir / 'result.json').read_text(encoding='utf-8'))


def read_progress(job_dir):
    """각 단계 스크립트가 20프레임마다 쓰는 progress.json을 읽는다. 아직 없거나 쓰는 중이라 깨졌으면 None."""
    try:
        return json.loads((Path(job_dir) / 'progress.json').read_text())
    except (OSError, ValueError):
        return None


if __name__ == '__main__':
    command, video, job = sys.argv[1:4]
    if command == 'people':
        found = find_people(video, job)
        print('people seen %d, candidates %d' % (found['people_seen'], len(found['candidates'])))
        for c in found['candidates']:
            print('  track %(track)d: largest %(max_px)dpx at %(best_seconds).1fs, analysable %(analysable_seconds).1fs' % c)
    else:
        result = analyse(video, job, int(sys.argv[4]))
        print(result['summary'], 'segments', result['segments'])
