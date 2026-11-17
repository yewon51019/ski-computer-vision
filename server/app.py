"""스키 자세 분석 웹 API (FastAPI).

화면(src/analysis.jsx)이 부르는 주소
  POST   /api/jobs                    영상 올리기. 올리자마자 뒤에서 '사람 찾기'가 시작된다
  GET    /api/jobs/{id}               작업 상태·진행률, 그리고 (찾은 뒤에는) 고를 수 있는 사람 후보
  GET    /api/jobs/{id}/thumb/{track} 후보 한 명의 그림
  POST   /api/jobs/{id}/select        {"track": n} 이 사람을 분석해 달라. 뒤에서 '관절 분석'이 시작된다
  GET    /api/jobs/{id}/result        분석 결과(result.json). 브라우저가 원본 영상 위에 그리는 좌표
  DELETE /api/jobs/{id}               서버에 올린 영상과 결과를 지운다

작업 상태: people(사람 찾는 중) → choose(고르기 대기) → pose(관절 분석 중) → done(끝), 문제가 생기면 error.
GPU가 하나라서 분석은 한 번에 하나씩만 돈다. 서버는 화면이 지워 달라고 할 때까지만 작업을 보관한다.
오래 보관하는 것은 사용자 기기의 몫이다(웹 DB를 쓰지 않기로 한 결정).

실행 (프로젝트 루트에서)
  .web-venv\\Scripts\\python.exe -m uvicorn server.app:app --port 8000
"""
import json
import re
import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# 프로젝트 루트를 import 경로에 넣어 루트의 ski_pipeline.py를 불러올 수 있게 한다
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ski_pipeline  # noqa: E402

JOBS = ROOT / 'data' / 'jobs'                   # 작업마다 data/jobs/<작업 ID>/ 폴더가 생긴다
MAX_BYTES = 300 * 1024 * 1024                   # 올리기 크기 한도 300MB (임시값, 사용자와 아직 정하지 않음)
ALLOWED = {'.mp4', '.mov'}                      # 받는 영상 형식
JOB_ID = re.compile(r'^[0-9a-f]{32}$')          # 작업 ID 모양(16진수 32자). 이 모양이 아니면 폴더 경로로 쓰지 않는다

app = FastAPI(title='Ski knee analysis')
# 개발 서버(5173번)에서 직접 부를 때를 위한 허용 목록. 보통은 vite가 /api를 대신 넘겨 준다
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
                   allow_methods=['*'], allow_headers=['*'])

gpu = threading.Lock()                 # GPU 사용 잠금: 분석은 한 번에 하나만
running, errors = {}, {}               # 작업 ID → 지금 하는 단계 / 오류 문구 (서버 메모리에만 있음)


def job_dir(job_id):
    """작업 ID로 작업 폴더를 찾는다. 모양이 틀리거나 없는 작업이면 404를 돌려준다.

    ID 모양을 먼저 검사해서 '../' 같은 값으로 다른 폴더에 접근하는 것을 막는다.
    """
    if not JOB_ID.match(job_id):
        raise HTTPException(404, 'Unknown job')
    path = JOBS / job_id
    if not path.is_dir():
        raise HTTPException(404, 'Unknown job')
    return path


def video_of(path):
    """작업 폴더 안의 올린 영상 파일(video.mp4 또는 video.mov)."""
    return next(p for p in path.iterdir() if p.stem == 'video')


def work(job_id, stage, function, *arguments):
    """뒤에서(별도 스레드) 분석 한 단계를 실행한다.

    실행 중에는 running에 단계 이름을 적어 두고, GPU 잠금을 잡은 채 실행한다.
    다른 분석이 GPU를 쓰는 중이면 끝날 때까지 기다린다. 실패하면 오류 문구(끝 500자)를
    errors에 남겨 화면이 보여 주게 한다.
    """
    running[job_id] = stage
    errors.pop(job_id, None)
    try:
        with gpu:
            function(*arguments)
    except Exception as error:                      # 어떤 오류든 화면에 작업 오류로 보여 준다
        errors[job_id] = str(error)[-500:]
    finally:
        running.pop(job_id, None)


@app.post('/api/jobs')
async def create_job(file: UploadFile = File(...)):
    """영상을 받아 새 작업을 만들고, 뒤에서 사람 찾기를 시작한다. 돌려주는 값: {"id": 작업 ID}."""
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, 'Upload an MP4 or MOV video')
    job_id = uuid.uuid4().hex
    path = JOBS / job_id
    path.mkdir(parents=True)
    video, size = path / ('video' + suffix), 0
    # 1MB씩 받아 쓰면서 크기를 센다. 한도를 넘는 순간 멈추고 폴더째 지운다 (큰 파일을 끝까지 받지 않음)
    with video.open('wb') as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                out.close()
                shutil.rmtree(path)
                raise HTTPException(413, 'Video is larger than %d MB' % (MAX_BYTES // 1024 // 1024))
            out.write(chunk)
    # daemon=True: 서버를 끄면 이 스레드도 같이 끝난다 (그 작업은 '다시 올려 주세요' 오류가 된다)
    threading.Thread(target=work, args=(job_id, 'people', ski_pipeline.find_people, video, path), daemon=True).start()
    return {'id': job_id}


@app.get('/api/jobs/{job_id}')
def job_state(job_id: str):
    """작업 상태를 알려 준다. 화면이 1초마다 부른다.

    상태 판단 순서: 오류가 있으면 error → 실행 중이면 그 단계 → result.json이 있으면 done
    → candidates.json이 있으면 choose → 아무것도 아니면(서버가 중간에 꺼진 경우) error.
    """
    path = job_dir(job_id)
    state = dict(id=job_id, progress=ski_pipeline.read_progress(path))
    if job_id in errors:
        state.update(state='error', error=errors[job_id])
    elif job_id in running:
        state['state'] = running[job_id]
        # waiting: 다른 작업이 GPU를 쓰고 있어 아직 시작을 못 한 상태 (진행 기록이 다른 단계 것)
        state['waiting'] = gpu.locked() and (state['progress'] or {}).get('stage') != running[job_id]
    elif (path / 'result.json').is_file():
        state['state'] = 'done'
    elif (path / 'candidates.json').is_file():
        state['state'] = 'choose'
    else:
        state.update(state='error', error='The analysis stopped before it finished. Upload the video again.')
    # 후보 목록이 생긴 뒤에는 항상 함께 보낸다 ('다른 사람 선택'으로 돌아갈 때도 쓰임)
    if (path / 'candidates.json').is_file():
        found = json.loads((path / 'candidates.json').read_text(encoding='utf-8'))
        state.update(video=found['video'], candidates=found['candidates'])
    return state


@app.get('/api/jobs/{job_id}/thumb/{track}')
def thumb(job_id: str, track: int):
    """후보 한 명의 그림(person_tracks.py가 만든 thumb_<track>.jpg)."""
    picture = job_dir(job_id) / ('thumb_%d.jpg' % track)
    if not picture.is_file():
        raise HTTPException(404, 'Unknown person')
    return FileResponse(picture, media_type='image/jpeg')


class Selection(BaseModel):
    """POST /select 요청 본문: 고른 사람의 추적 번호."""
    track: int


@app.post('/api/jobs/{job_id}/select')
def select(job_id: str, selection: Selection):
    """고른 사람의 관절 분석을 뒤에서 시작한다. 같은 작업에서 다른 사람을 다시 고를 수도 있다."""
    path = job_dir(job_id)
    if job_id in running:
        raise HTTPException(409, 'This video is still being analysed')
    if not (path / 'candidates.json').is_file():
        raise HTTPException(409, 'People have not been found yet')
    found = json.loads((path / 'candidates.json').read_text(encoding='utf-8'))
    if selection.track not in {c['track'] for c in found['candidates']}:
        raise HTTPException(422, 'Pick one of the listed people')
    (path / 'result.json').unlink(missing_ok=True)       # 이전에 고른 사람의 결과는 지운다
    threading.Thread(target=work, args=(job_id, 'pose', ski_pipeline.analyse, video_of(path), path, selection.track),
                     daemon=True).start()
    return {'id': job_id, 'state': 'pose'}


@app.get('/api/jobs/{job_id}/result')
def result(job_id: str):
    """분석 결과 JSON. 분석이 끝나지 않았으면 409."""
    file = job_dir(job_id) / 'result.json'
    if not file.is_file() or job_id in running:
        raise HTTPException(409, 'The result is not ready')
    return JSONResponse(json.loads(file.read_text(encoding='utf-8')))


@app.delete('/api/jobs/{job_id}')
def delete(job_id: str):
    """작업 폴더(영상·결과)를 통째로 지운다. 분석 중에는 지울 수 없다(409)."""
    path = job_dir(job_id)
    if job_id in running:
        raise HTTPException(409, 'This video is still being analysed')
    shutil.rmtree(path)
    errors.pop(job_id, None)
    return {'deleted': job_id}
