#!/usr/bin/env python3
import argparse, asyncio, json, logging, os, ssl, time, threading
from collections import deque
from pathlib import Path
from fractions import Fraction

import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger('maturai')
VIDEO_CLOCK = 90000
TIME_BASE = Fraction(1, VIDEO_CLOCK)

class CameraBuffer:
    def __init__(self, cam_id, delay_ms=400, fps=5, seconds=3):
        self.cam_id = cam_id
        self.delay = delay_ms / 1000
        self.fps = fps
        self.frames = deque(maxlen=max(12, int(fps * seconds * 2)))
        self.condition = asyncio.Condition()
        self.last_seen = 0.0
        self.width = self.height = 0
        self.received = 0
        self.source_pc = None
        self.processor = None
        self.detections = []
        self.detected_at = 0.0
        self.detect_ms = 0.0
        self.detect_fps = 0.0
        self.detect_task = None

    async def ingest(self, track, owner):
        LOG.info('camera %s: recepcao iniciada', self.cam_id)
        try:
            while True:
                frame = await track.recv()
                now = time.monotonic()
                if self.source_pc is not owner:
                    continue
                changed = (self.width, self.height) != (frame.width, frame.height)
                self.width, self.height = frame.width, frame.height
                if changed:
                    LOG.warning('camera %s: frame recebido pelo aiortc mudou para %sx%s', self.cam_id, frame.width, frame.height)
                self.last_seen = now
                self.received += 1
                async with self.condition:
                    self.frames.append((now, frame))
                    self.condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.info('camera %s: fim da trilha (%s)', self.cam_id, exc)

    def choose(self):
        if not self.frames:
            return None
        target = time.monotonic() - self.delay
        chosen = self.frames[0][1]
        for received_at, frame in self.frames:
            if received_at <= target:
                chosen = frame
            else:
                break
        return chosen

    def online(self):
        return self.last_seen and (time.monotonic() - self.last_seen < 4.0)

class BufferedCameraTrack(MediaStreamTrack):
    kind = 'video'
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.started = time.monotonic()
        self.index = 0

    async def recv(self):
        interval = 1 / self.camera.fps
        due = self.started + self.index * interval
        await asyncio.sleep(max(0, due - time.monotonic()))
        deadline = time.monotonic() + 2
        frame = self.camera.choose()
        while frame is None and time.monotonic() < deadline:
            async with self.camera.condition:
                try:
                    await asyncio.wait_for(self.camera.condition.wait(), 0.5)
                except asyncio.TimeoutError:
                    pass
            frame = self.camera.choose()
        if frame is None:
            img = __import__('numpy').zeros((540, 960, 3), dtype='uint8')
            frame = VideoFrame.from_ndarray(img, format='bgr24')
        else:
            # Rebuild from ndarray so every viewer owns a stable frame and YOLO can use BGR directly.
            frame = VideoFrame.from_ndarray(frame.to_ndarray(format='bgr24'), format='bgr24')
        frame.pts = int(self.index * VIDEO_CLOCK / self.camera.fps)
        frame.time_base = TIME_BASE
        self.index += 1
        return frame

class YoloDetector:
    def __init__(self, model_path, device='0', imgsz=640, conf=0.35):
        if YOLO is None:
            raise RuntimeError('ultralytics nao instalado: pip install ultralytics')
        self.model_path = model_path
        self.device = device
        self.requested_imgsz = imgsz
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(model_path)
        self.lock = threading.Lock()
        self.onnx_input_shape = None
        if Path(model_path).suffix.lower() == '.onnx':
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
                shape = session.get_inputs()[0].shape
                self.onnx_input_shape = shape
                height, width = shape[-2], shape[-1]
                if isinstance(height, int) and isinstance(width, int):
                    self.imgsz = (height, width)
                    if imgsz not in (height, width):
                        LOG.warning('ONNX possui entrada fixa %sx%s; ignorando --imgsz=%s', width, height, imgsz)
                else:
                    LOG.info('ONNX possui entrada dinamica: %s', shape)
            except Exception as exc:
                LOG.warning('Nao foi possivel inspecionar a entrada ONNX (%s); usando --imgsz=%s', exc, imgsz)
        LOG.info('YOLO carregado: %s | device=%s | imgsz efetivo=%s', model_path, device, self.imgsz)

    def predict(self, image):
        started = time.perf_counter()
        with self.lock:
            results = self.model.predict(source=image, imgsz=self.imgsz, conf=self.conf,
                                         device=self.device, verbose=False)
        detections = []
        if results:
            r = results[0]
            h, w = image.shape[:2]
            names = getattr(r, 'names', {}) or {}
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.detach().cpu().numpy()
                confs = r.boxes.conf.detach().cpu().numpy()
                classes = r.boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, cls in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = map(float, box)
                    detections.append({
                        'x1': max(0.0, min(1.0, x1 / w)),
                        'y1': max(0.0, min(1.0, y1 / h)),
                        'x2': max(0.0, min(1.0, x2 / w)),
                        'y2': max(0.0, min(1.0, y2 / h)),
                        'confidence': float(score),
                        'class_id': int(cls),
                        'label': str(names.get(int(cls), 'face')),
                    })
        return detections, (time.perf_counter() - started) * 1000

class Hub:
    def __init__(self, delay_ms, fps, detector=None, detect_fps=10, debug=False):
        self.delay_ms, self.fps = delay_ms, fps
        self.debug = debug
        self.client_debug = deque(maxlen=500)
        self.detector = detector
        self.detect_fps = max(1, min(60, detect_fps))
        self.cameras = {}
        self.pcs = set()
        self.tasks = set()

    def camera(self, cam_id):
        if cam_id not in self.cameras:
            self.cameras[cam_id] = CameraBuffer(cam_id, self.delay_ms, self.fps)
        return self.cameras[cam_id]

    def task(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self.tasks.discard)
        return t

    async def detect_loop(self, cam):
        interval = 1 / self.detect_fps
        last_count = 0
        LOG.info('camera %s: detector iniciado a ate %s FPS', cam.cam_id, self.detect_fps)
        while True:
            started = time.monotonic()
            try:
                if cam.frames and cam.received != last_count:
                    last_count = cam.received
                    frame = cam.frames[-1][1]
                    image = frame.to_ndarray(format='bgr24')
                    detections, elapsed_ms = await asyncio.to_thread(self.detector.predict, image)
                    now = time.monotonic()
                    previous = cam.detected_at
                    cam.detections = detections
                    cam.detect_ms = elapsed_ms
                    cam.detected_at = now
                    if previous:
                        instant = 1 / max(0.001, now - previous)
                        cam.detect_fps = instant if not cam.detect_fps else cam.detect_fps * 0.8 + instant * 0.2
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception('camera %s: erro temporario no detector; tentando novamente', cam.cam_id)
                cam.detections = []
                await asyncio.sleep(0.5)
            await asyncio.sleep(max(0.001, interval - (time.monotonic() - started)))

    async def page(self, request):
        name = request.match_info['name']
        return web.FileResponse(ROOT / f'{name}.html', headers={'Cache-Control':'no-store'})

    async def index(self, request):
        raise web.HTTPFound('/viewer')

    async def offer_camera(self, request):
        data = await request.json()
        cam_id = str(data.get('camera_id', '')).strip()
        if not cam_id or len(cam_id) > 32 or not all(c.isalnum() or c in '-_' for c in cam_id):
            raise web.HTTPBadRequest(text='camera_id invalido')
        pc = RTCPeerConnection(); self.pcs.add(pc)
        cam = self.camera(cam_id)
        old = cam.source_pc

        @pc.on('track')
        def on_track(track):
            if track.kind == 'video':
                self.task(cam.ingest(track, pc))
                if self.detector and (cam.detect_task is None or cam.detect_task.done()):
                    cam.detect_task = self.task(self.detect_loop(cam))

        @pc.on('connectionstatechange')
        async def state():
            LOG.info('camera %s: PC=%s ICE=%s', cam_id, pc.connectionState, pc.iceConnectionState)
            if pc.connectionState == 'connected':
                cam.source_pc = pc
                if old and old is not pc:
                    await old.close()
                    self.pcs.discard(old)
            elif pc.connectionState in ('failed', 'closed'):
                if cam.source_pc is pc:
                    cam.source_pc = None
                if pc.connectionState != 'closed':
                    await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
        answer = await pc.createAnswer(); await pc.setLocalDescription(answer)
        return web.json_response({'sdp':pc.localDescription.sdp,'type':pc.localDescription.type})

    async def offer_viewer(self, request):
        data = await request.json(); cam_id = str(data.get('camera_id','')).strip()
        cam = self.cameras.get(cam_id)
        if not cam:
            raise web.HTTPNotFound(text='camera inexistente')
        pc = RTCPeerConnection(); self.pcs.add(pc)
        pc.addTrack(BufferedCameraTrack(cam))

        @pc.on('connectionstatechange')
        async def state():
            if pc.connectionState in ('failed','closed'):
                if pc.connectionState != 'closed':
                    await pc.close()
                self.pcs.discard(pc)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
        answer = await pc.createAnswer(); await pc.setLocalDescription(answer)
        return web.json_response({'sdp':pc.localDescription.sdp,'type':pc.localDescription.type})

    async def status(self, request):
        now = time.monotonic()
        cams=[]
        for cid, c in sorted(self.cameras.items()):
            cams.append({'id':cid,'online':c.online(),'age_ms':round((now-c.last_seen)*1000) if c.last_seen else None,
                         'width':c.width,'height':c.height,'received':c.received,'buffered':len(c.frames),
                         'delay_ms':self.delay_ms,'output_fps':self.fps,
                         'detections':len(c.detections),'detect_ms':round(c.detect_ms,1),
                         'detect_fps':round(c.detect_fps,1)})
        return web.json_response({'cameras':cams})

    async def detections(self, request):
        cam = self.cameras.get(request.match_info['cam_id'])
        if not cam:
            raise web.HTTPNotFound(text='camera inexistente')
        age_ms = round((time.monotonic() - cam.detected_at) * 1000) if cam.detected_at else None
        return web.json_response({
            'camera_id': cam.cam_id,
            'width': cam.width, 'height': cam.height,
            'detections': cam.detections,
            'age_ms': age_ms,
            'inference_ms': round(cam.detect_ms, 1),
            'detect_fps': round(cam.detect_fps, 1),
        }, headers={'Cache-Control':'no-store'})

    async def debug_config(self, request):
        return web.json_response({'enabled': self.debug}, headers={'Cache-Control':'no-store'})

    async def debug_client(self, request):
        if not self.debug:
            raise web.HTTPNotFound()
        data = await request.json()
        entry = {'time': time.time(), 'camera_id': str(data.get('camera_id','?')),
                 'event': str(data.get('event','client')), 'data': data.get('data')}
        self.client_debug.append(entry)
        LOG.warning('BROWSER DEBUG cam=%s event=%s data=%s', entry['camera_id'], entry['event'], entry['data'])
        return web.json_response({'ok':True})

    async def debug_dump(self, request):
        if not self.debug:
            raise web.HTTPNotFound()
        return web.json_response({'events':list(self.client_debug)}, headers={'Cache-Control':'no-store'})

    async def snapshot(self, request):
        cam = self.cameras.get(request.match_info['cam_id'])
        frame = cam.choose() if cam else None
        if frame is None: raise web.HTTPNotFound(text='sem frame')
        ok, jpg = cv2.imencode('.jpg', frame.to_ndarray(format='bgr24'), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok: raise web.HTTPInternalServerError()
        return web.Response(body=jpg.tobytes(), content_type='image/jpeg', headers={'Cache-Control':'no-store'})


    async def mjpeg(self, request):
        cam = self.cameras.get(request.match_info['cam_id'])
        if not cam:
            raise web.HTTPNotFound(text='camera inexistente')
        fps = max(1, min(30, int(request.query.get('fps', cam.fps))))
        quality = max(30, min(92, int(request.query.get('q', 76))))
        bitrate_kbps = max(0, min(20000, int(request.query.get('bitrate_kbps', 0))))
        target_bytes = int((bitrate_kbps * 1000) / 8 / fps) if bitrate_kbps else 0

        def encode_jpeg(img):
            if not target_bytes:
                return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            # Best effort bitrate limiter for MJPEG: binary search JPEG quality for target bytes/frame.
            low, high = 30, 92
            best = None
            for _ in range(6):
                mid = (low + high) // 2
                ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, mid])
                if not ok:
                    return ok, jpg
                best = (ok, jpg)
                if len(jpg) > target_bytes:
                    high = mid - 1
                else:
                    low = mid + 1
            return best

        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        })
        await response.prepare(request)
        interval = 1 / fps
        try:
            while True:
                started = time.monotonic()
                frame = cam.choose()
                if frame is not None:
                    img = frame.to_ndarray(format='bgr24')
                    ok, jpg = encode_jpeg(img)
                    if ok:
                        body = jpg.tobytes()
                        await response.write(b'--frame\r\n')
                        await response.write(b'Content-Type: image/jpeg\r\n')
                        await response.write(f'Content-Length: {len(body)}\r\n\r\n'.encode())
                        await response.write(body + b'\r\n')
                await asyncio.sleep(max(0.001, interval - (time.monotonic() - started)))
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:
            LOG.info('mjpeg %s finalizado: %s', cam.cam_id, exc)
        return response

    async def shutdown(self, app):
        for t in list(self.tasks): t.cancel()
        await asyncio.gather(*(pc.close() for pc in list(self.pcs)), return_exceptions=True)

    def app(self):
        app=web.Application(client_max_size=2*1024*1024)
        app.router.add_get('/', self.index)
        app.router.add_get('/{name:camera|viewer}', self.page)
        app.router.add_post('/api/offer/camera', self.offer_camera)
        app.router.add_post('/api/offer/viewer', self.offer_viewer)
        app.router.add_get('/api/status', self.status)
        app.router.add_get('/api/debug/config', self.debug_config)
        app.router.add_post('/api/debug/client', self.debug_client)
        app.router.add_get('/api/debug/dump', self.debug_dump)
        app.router.add_get('/api/frame/{cam_id}', self.snapshot)
        app.router.add_get('/api/detections/{cam_id}', self.detections)
        app.router.add_get('/api/mjpeg/{cam_id}', self.mjpeg)
        app.on_shutdown.append(self.shutdown)
        return app

def tls_context(cert, key):
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.minimum_version=ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert, key); return ctx

def detect_cert(root):
    certs=list(root.glob('*.crt')) + list(root.glob('*.pem'))
    keys=list(root.glob('*.key'))
    for cert in certs:
        stem=cert.name.removesuffix(cert.suffix)
        for key in keys:
            if key.stem == stem: return str(cert), str(key)
    return (str(certs[0]),str(keys[0])) if certs and keys else (None,None)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='MaturAI WebRTC Hub')
    p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8443)
    p.add_argument('--fps',type=int,default=5); p.add_argument('--delay-ms',type=int,default=200)
    p.add_argument('--cert'); p.add_argument('--key'); p.add_argument('-v','--verbose',action='store_true')
    p.add_argument('--debug', action='store_true', help='Ativa console visual e telemetria detalhada do navegador/WebRTC')
    p.add_argument('--model', required=True, help='Caminho do yolov8n-face.pt ou .onnx')
    p.add_argument('--device', default='0', help='0 para GPU NVIDIA; cpu para CPU')
    p.add_argument('--imgsz', type=int, default=640, help='Tamanho de inferencia. Nao e a resolucao da camera; ONNX fixo ignora valor incompatível')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--detect-fps', type=int, default=10)
    a=p.parse_args(); logging.basicConfig(level=logging.DEBUG if (a.verbose or a.debug) else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cert,key=(a.cert,a.key) if a.cert and a.key else detect_cert(ROOT)
    if not cert: raise SystemExit('Certificado/chave nao encontrados. Use --cert ARQ.crt --key ARQ.key')
    detector=YoloDetector(a.model, a.device, a.imgsz, a.conf)
    LOG.info('MaturAI em https://nitro.tail35309f.ts.net:%s/viewer',a.port)
    web.run_app(Hub(a.delay_ms,a.fps,detector,a.detect_fps,a.debug).app(),host=a.host,port=a.port,ssl_context=tls_context(cert,key),access_log=None)
