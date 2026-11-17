"""Render one EPFL Ski 2DPose annotation as an offline HTML overlay."""
import argparse
import base64
import json
import math
from html import escape
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def knee_angle(points, start):
    hip, knee, ankle = points[start:start + 3]
    if not all(p[2] for p in (hip, knee, ankle)):
        return None
    ax, ay = hip[0] - knee[0], hip[1] - knee[1]
    bx, by = ankle[0] - knee[0], ankle[1] - knee[1]
    if math.hypot(ax, ay) == 0 or math.hypot(bx, by) == 0:
        return None
    return math.degrees(math.atan2(abs(ax * by - ay * bx), ax * bx + ay * by))


def webp_size(data):
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        raise ValueError('Invalid WebP file')
    kind = data[12:16]
    if kind == b'VP8X':
        return tuple(1 + int.from_bytes(data[i:i+3], 'little') for i in (24, 27))
    if kind == b'VP8L':
        bits = int.from_bytes(data[21:25], 'little')
        return (bits & 16383) + 1, ((bits >> 14) & 16383) + 1
    if kind == b'VP8 ':
        return tuple(int.from_bytes(data[i:i+2], 'little') & 16383 for i in (26, 28))
    raise ValueError('Unknown WebP header')


def render(photo, record, title):
    width, height = webp_size(photo)
    points = [(x * width, y * height, visible) for x, y, visible in record['annotation']]
    if len(points) != 24:
        raise ValueError('Expected 24 EPFL keypoints')
    encoded = base64.b64encode(photo).decode('ascii')
    shapes = [f'<image width="{width}" height="{height}" href="data:image/webp;base64,{encoded}"/>']
    for chain in ((0,1,2,3,4,5),(1,6,7,8,9),(2,10,11,12),(6,13,14,15),
                  (10,13),(16,17,18,19),(20,21,22,23)):
        for a, b in zip(chain, chain[1:]):
            x, y, visible = points[a]
            u, v, other_visible = points[b]
            if visible and other_visible:
                shapes.append(f'<line x1="{x}" y1="{y}" x2="{u}" y2="{v}" stroke="cyan" stroke-width="3"/>')
    for idx, (x, y, visible) in enumerate(points):
        if visible:
            shapes.append(f'<circle cx="{x}" cy="{y}" r="4" fill="gold" stroke="black"><title>{idx}</title></circle>')
    measurements = []
    for side, start in [('오른쪽', 10), ('왼쪽', 13)]:
        angle = knee_angle(points, start)
        value = '측정 불가' if angle is None else f'{angle:.1f}°'
        measurements.append(f'{side} 무릎 {value}')
    caption = escape(f"{title} · 프레임 {record['frame_idx']}")
    return (f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>스키 자세</title>'
            f'<h1>스키 자세 · 정답 좌표</h1><p>{caption}</p>'
            f'<svg style="width:100%;max-height:75vh" viewBox="0 0 {width} {height}">{"".join(shapes)}</svg>'
            f'<p>{" · ".join(measurements)} (영상 평면 내각)</p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', default='ski2dpose_labels.json.zip')
    parser.add_argument('--images', default='ski2dpose_images_webp.zip')
    parser.add_argument('--id', nargs=3, metavar=('VIDEO', 'SPLIT', 'IMAGE'))
    parser.add_argument('--output', type=Path, default=Path('ski_pose.html'))
    args = parser.parse_args()
    with ZipFile(args.labels) as archive:
        names = [n for n in archive.namelist() if n.endswith('.json') and '__MACOSX/' not in n]
        if len(names) != 1:
            parser.error('Expected one labels JSON in the archive')
        labels = json.loads(archive.read(names[0]))
    with ZipFile(args.images) as archive:
        paths = {}
        for name in archive.namelist():
            path = PurePosixPath(name)
            if path.suffix.lower() == '.webp' and '__MACOSX' not in path.parts:
                key = (path.parent.parent.name, path.parent.name, path.stem)
                if key in paths:
                    parser.error(f'Ambiguous image path: {key}')
                paths[key] = name
        for video, splits in labels.items():
            for split, records in splits.items():
                for image, record in records.items():
                    ids = (video, split, image)
                    if args.id and list(ids) != args.id:
                        continue
                    key = (video, split, PurePosixPath(image).stem)
                    if key in paths:
                        html = render(archive.read(paths[key]), record, ' / '.join(ids))
                        args.output.write_text(html, encoding='utf-8')
                        print(args.output.resolve())
                        return
    parser.error(f'No matching WebP. Example archive key: {next(iter(paths), None)}')


if __name__ == '__main__':
    main()
