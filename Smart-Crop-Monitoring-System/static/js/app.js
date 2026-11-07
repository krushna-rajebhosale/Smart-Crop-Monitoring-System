/* Smart Crop Monitoring System - page script (no inline JS, so the
   Content-Security-Policy can stay strict).

   Home page: two ways to give a photo
     1. "Take photo": the camera opens by itself, shows a live preview result,
        and the round button takes the photo and sends it for the full report.
     2. "Upload photo": choose or drag a file. */
(function () {
  'use strict';

  // ---- Settings you may change --------------------------------------------
  var OPEN_CAMERA_ON_LOAD = true;   // false = start on the "Upload photo" tab instead
  var LIVE_PREVIEW = true;          // false = camera only, no live result while you look
  var LIVE_DELAY_MS = 1500;         // pause between two live checks (1500 = 1.5 seconds)
  // --------------------------------------------------------------------------

  var MAX_BYTES = 8 * 1024 * 1024;
  var ALLOWED = ['image/jpeg', 'image/png'];

  var $ = function (id) { return document.getElementById(id); };

  var form = $('uploadForm');
  if (form) {
    var input = $('leaf_image');
    var box = $('dropzoneBox');
    var uploadText = $('uploadText');
    var uploadHint = $('uploadHint');
    var submitBtn = $('submitBtn');
    var previewPanel = $('previewPanel');
    var previewImage = $('previewImage');
    var previewUrl = null;

    var tabCamera = $('tabCamera');
    var tabUpload = $('tabUpload');
    var cameraMode = $('cameraMode');
    var uploadMode = $('uploadMode');
    var camWrap = $('camWrap');
    var camStage = $('camStage');
    var camUnsupported = $('camUnsupported');
    var video = $('cameraVideo');
    var hud = $('liveHud');
    var hudText = $('liveStatus');
    var liveResult = $('liveResult');
    var liveName = $('liveName');
    var liveCrop = $('liveCrop');
    var livePercent = $('livePercent');
    var liveBar = $('liveBar');
    var shutterBtn = $('shutterBtn');
    var flipBtn = $('flipBtn');
    var pauseBtn = $('pauseBtn');
    var phoneCameraBtn = $('phoneCameraBtn');
    var cameraFallback = $('cameraFallback');
    var cameraError = $('cameraError');

    var stream = null;
    var facing = 'environment';
    var running = false;      // live checks switched on
    var paused = false;
    var busy = false;
    var timer = null;
    var mode = 'camera';

    /* ------------------------------------------------------------ messages */
    var showCameraError = function (message) {
      cameraError.textContent = message;
      cameraError.hidden = false;
    };
    var clearCameraError = function () { cameraError.hidden = true; };

    var setHud = function (state, text) {
      hud.setAttribute('data-state', state);
      hudText.textContent = text;
    };

    /* -------------------------------------------------------- upload mode */
    var showError = function (title, hint) {
      input.value = '';
      box.classList.remove('has-file');
      box.classList.add('has-error');
      uploadText.textContent = title;
      uploadHint.textContent = hint;
      previewPanel.hidden = true;
    };

    input.addEventListener('change', function () {
      var file = input.files && input.files[0];
      box.classList.remove('has-file', 'has-error', 'is-dragover');
      if (!file) { previewPanel.hidden = true; return; }

      var nameOk = /\.(jpe?g|png)$/i.test(file.name);
      if (!nameOk || (file.type && ALLOWED.indexOf(file.type) === -1)) {
        showError('That file type is not supported', 'Choose a JPG, JPEG or PNG image.');
        return;
      }
      if (file.size > MAX_BYTES) {
        showError('That file is too large', 'Choose an image under 8 MB.');
        return;
      }

      box.classList.add('has-file');
      uploadText.textContent = file.name;
      uploadHint.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB, ready to analyse';

      if (previewUrl) { URL.revokeObjectURL(previewUrl); }
      previewUrl = URL.createObjectURL(file);
      previewImage.src = previewUrl;
      previewPanel.hidden = (mode !== 'upload');
    });

    ['dragenter', 'dragover'].forEach(function (name) {
      box.addEventListener(name, function () { box.classList.add('is-dragover'); });
    });
    ['dragleave', 'drop'].forEach(function (name) {
      box.addEventListener(name, function () { box.classList.remove('is-dragover'); });
    });

    /* ------------------------------------------------------- camera basics */
    var cameraSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

    var stopLive = function () {
      running = false;
      if (timer) { clearTimeout(timer); timer = null; }
    };

    var stopCamera = function () {
      stopLive();
      if (stream) {
        stream.getTracks().forEach(function (track) { track.stop(); });
        stream = null;
      }
      video.srcObject = null;
      shutterBtn.disabled = true;
    };

    // Cut out exactly the part of the video you can SEE (the box crops the picture),
    // so the photo matches what is on the screen.
    var grabFrame = function (maxWidth) {
      var vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) { return null; }
      var sw = camStage.clientWidth || vw, sh = camStage.clientHeight || vh;
      var target = sw / sh;
      var cw, ch;
      if (vw / vh > target) { ch = vh; cw = vh * target; } else { cw = vw; ch = vw / target; }
      var sx = (vw - cw) / 2, sy = (vh - ch) / 2;
      var scale = Math.min(1, maxWidth / cw);
      var canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(cw * scale));
      canvas.height = Math.max(1, Math.round(ch * scale));
      canvas.getContext('2d').drawImage(video, sx, sy, cw, ch, 0, 0, canvas.width, canvas.height);
      return canvas;
    };

    // Put a photo into the normal upload field, exactly as if the user had chosen a file.
    var setInputFile = function (file) {
      var transfer = new DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      input.dispatchEvent(new Event('change'));
    };

    var submitPhoto = function () {
      if (input.files && input.files.length) {
        form.requestSubmit();
      }
    };

    /* ---------------------------------------------------------- live check */
    var isHealthy = function (label) { return /___healthy$/i.test(label || ''); };

    var showLiveResult = function (data) {
      var state = data.low_confidence ? 'uncertain' : (isHealthy(data.prediction) ? 'healthy' : 'disease');
      var pct = Math.round(data.confidence);
      liveName.textContent = data.condition;
      liveCrop.textContent = data.crop ? ' \u00b7 ' + data.crop : '';
      livePercent.textContent = pct + '%';
      liveBar.value = data.confidence;
      liveResult.setAttribute('data-state', state);
      liveResult.hidden = false;
      if (state === 'uncertain') {
        setHud('uncertain', 'Not sure yet \u2013 hold steady');
      } else {
        setHud(state, data.condition + ' \u00b7 ' + pct + '%');
      }
    };

    var scheduleNext = function (delay) {
      if (!running) { return; }
      timer = setTimeout(liveStep, delay || LIVE_DELAY_MS);
    };

    var liveStep = function () {
      if (!running || paused || busy || document.hidden) { scheduleNext(); return; }
      var canvas = grabFrame(640);
      if (!canvas) { scheduleNext(500); return; }
      busy = true;
      canvas.toBlob(function (blob) {
        if (!blob) { busy = false; scheduleNext(); return; }
        var body = new FormData();
        body.append('leaf_image', blob, 'frame.jpg');
        fetch('/api/predict', { method: 'POST', body: body })
          .then(function (response) {
            return response.json().then(function (data) { return { status: response.status, data: data }; });
          })
          .then(function (reply) {
            busy = false;
            if (!running) { return; }
            if (reply.status === 200) {
              showLiveResult(reply.data);
              scheduleNext();
            } else if (reply.status === 400 && reply.data.code === 'not_leaf') {
              liveResult.hidden = true;
              setHud('notleaf', 'No leaf in view \u2013 move closer');
              scheduleNext();
            } else if (reply.status === 429) {
              setHud('scanning', 'Slowing down for a moment\u2026');
              scheduleNext(6000);
            } else if (reply.status === 503) {
              liveResult.hidden = true;
              setHud('error', 'The analysis service is not ready');
              stopLive();
            } else {
              liveResult.hidden = true;
              setHud('scanning', reply.data.error || 'Looking for a leaf\u2026');
              scheduleNext();
            }
          })
          .catch(function () {
            busy = false;
            setHud('error', 'Connection problem \u2013 retrying');
            scheduleNext(3000);
          });
      }, 'image/jpeg', 0.85);
    };

    var startLive = function () {
      if (!LIVE_PREVIEW) { setHud('idle', 'Camera ready'); return; }
      running = true;
      paused = false;
      pauseBtn.querySelector('span').textContent = 'Pause';
      pauseBtn.querySelector('i').className = 'fa-solid fa-pause';
      setHud('scanning', 'Looking for a leaf\u2026');
      scheduleNext(800);
    };

    /* --------------------------------------------------------- open camera */
    var startCamera = function () {
      clearCameraError();
      stopCamera();

      if (!cameraSupported) {
        camWrap.hidden = true;
        camUnsupported.hidden = false;
        return;
      }
      camUnsupported.hidden = true;
      camWrap.hidden = false;
      liveResult.hidden = true;
      setHud('idle', 'Starting camera\u2026');

      navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false
      }).then(function (mediaStream) {
        if (mode !== 'camera') {            // user already left the tab
          mediaStream.getTracks().forEach(function (t) { t.stop(); });
          return;
        }
        stream = mediaStream;
        video.srcObject = mediaStream;
        return video.play().then(function () {
          shutterBtn.disabled = false;
          startLive();
          navigator.mediaDevices.enumerateDevices().then(function (devices) {
            var cameras = devices.filter(function (d) { return d.kind === 'videoinput'; });
            flipBtn.hidden = cameras.length < 2;
          }).catch(function () {});
        });
      }).catch(function (error) {
        stopCamera();
        var message;
        if (error && error.name === 'NotAllowedError') {
          message = 'Camera permission was blocked. Allow the camera for this site in your browser (the camera icon next to the address), then press "Take photo" again.';
        } else if (error && error.name === 'NotFoundError') {
          message = 'No camera was found on this device. You can upload a photo instead.';
        } else {
          message = 'The camera could not be started. Close other apps that use the camera (Zoom, Teams) and try again, or upload a photo instead.';
        }
        showCameraError(message);
        setMode('upload', true);
      });
    };

    /* ----------------------------------------------------------- switch tab */
    var setMode = function (newMode, keepError) {
      mode = newMode;
      var camera = (newMode === 'camera');
      tabCamera.classList.toggle('is-active', camera);
      tabUpload.classList.toggle('is-active', !camera);
      tabCamera.setAttribute('aria-selected', camera ? 'true' : 'false');
      tabUpload.setAttribute('aria-selected', camera ? 'false' : 'true');
      cameraMode.hidden = !camera;
      uploadMode.hidden = camera;
      submitBtn.hidden = camera;
      previewPanel.hidden = camera || !(input.files && input.files.length && box.classList.contains('has-file'));
      if (!keepError) { clearCameraError(); }
      if (camera) { startCamera(); } else { stopCamera(); }
    };

    tabCamera.addEventListener('click', function () { if (mode !== 'camera') { setMode('camera'); } });
    tabUpload.addEventListener('click', function () { if (mode !== 'upload') { setMode('upload'); } });

    /* ------------------------------------------------------------- buttons */
    shutterBtn.addEventListener('click', function () {
      var canvas = grabFrame(1600);
      if (!canvas) { showCameraError('The camera is not ready yet. Wait a moment and try again.'); return; }
      shutterBtn.disabled = true;
      setHud('scanning', 'Analysing photo\u2026');
      canvas.toBlob(function (blob) {
        if (!blob) {
          shutterBtn.disabled = false;
          showCameraError('The photo could not be captured. Please try again.');
          return;
        }
        stopLive();
        setInputFile(new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' }));
        submitPhoto();
      }, 'image/jpeg', 0.92);
    });

    pauseBtn.addEventListener('click', function () {
      paused = !paused;
      pauseBtn.querySelector('span').textContent = paused ? 'Resume' : 'Pause';
      pauseBtn.querySelector('i').className = paused ? 'fa-solid fa-play' : 'fa-solid fa-pause';
      if (paused) { setHud('idle', 'Live check paused'); } else { setHud('scanning', 'Looking for a leaf\u2026'); }
    });

    flipBtn.addEventListener('click', function () {
      facing = (facing === 'environment') ? 'user' : 'environment';
      startCamera();
    });

    phoneCameraBtn.addEventListener('click', function () { cameraFallback.click(); });
    cameraFallback.addEventListener('change', function () {
      if (cameraFallback.files && cameraFallback.files[0]) {
        setInputFile(cameraFallback.files[0]);
        submitPhoto();
      }
    });

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) { setHud('idle', 'Live check paused'); }
    });
    window.addEventListener('pagehide', stopCamera);

    /* --------------------------------------------------------- send the form */
    form.addEventListener('submit', function (event) {
      if (!input.files || !input.files.length) {
        event.preventDefault();
        showError('Please choose a leaf image first', 'JPG, JPEG or PNG, up to 8 MB');
        return;
      }
      stopCamera();
      submitBtn.classList.add('is-loading');
      submitBtn.querySelector('i').className = 'fa-solid fa-spinner';
      submitBtn.querySelector('span').textContent = 'Analysing\u2026';
      setTimeout(function () { submitBtn.disabled = true; }, 0);
    });

    // Reset when the user comes back with the browser's Back button.
    window.addEventListener('pageshow', function (e) {
      if (e.persisted) { window.location.reload(); }
    });

    /* ----------------------------------------------------------------- start */
    var hadError = form.getAttribute('data-has-error') === '1';
    if (OPEN_CAMERA_ON_LOAD && !hadError) {
      setMode('camera');
    } else {
      setMode('upload', true);
    }
  }

  var printButtons = document.querySelectorAll('[data-print]');
  Array.prototype.forEach.call(printButtons, function (btn) {
    btn.addEventListener('click', function () { window.print(); });
  });
})();