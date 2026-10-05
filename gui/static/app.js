"use strict";

// ------------------------------------------------------------------ helpers

const $ = (sel, root = document) => root.querySelector(sel);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || res.statusText);
  return body;
}
const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 3500);
}

const basename = (p) => p.split("/").pop();

// ------------------------------------------------------------------ parameters

const TRACK_FIELDS = [
  { key: "gpus", label: "GPUs", type: "gpus", help: "Points are split across the selected GPUs." },
  { key: "mode", label: "Mode", type: "select", options: ["offline", "online"],
    help: "offline: whole clip at once (most accurate). online: sliding window, for long videos." },
  { key: "grid_size", label: "Grid size", type: "text", help: '"30" = 30×30 points, or COLSxROWS like "100x50".' },
  { key: "radius", label: "Dot radius (px)", type: "number", min: 0 },
  { key: "chunk_size", label: "Chunk size", type: "number", min: 0,
    help: "Max points per forward pass; lower it on out-of-memory. 0 = split evenly over GPUs." },
  { key: "frame_stride", label: "Frame stride", type: "number", min: 1, help: "Use every k-th frame." },
  { key: "max_frames", label: "Max frames", type: "number", min: 0, help: "0 = all frames." },
  { key: "grid_query_frame", label: "Grid start frame", type: "number", min: 0 },
  { key: "backward_tracking", label: "Backward tracking", type: "checkbox", help: "Offline only." },
  { key: "mask", label: "Mask image", type: "mask", help: "Optional: keep only grid points inside the mask." },
];
const TRACK_DEFAULTS = {
  gpus: [0], mode: "offline", grid_size: "30", radius: 2, chunk_size: 0, frame_stride: 1,
  max_frames: 0, grid_query_frame: 0, backward_tracking: false, mask: "",
};

const COMPARE_FIELDS = [
  { key: "view", label: "View", type: "select", options: ["pairs", "diff"],
    help: "pairs: green = point has a visible pair in B, gray = no pair. diff: color by |A − B|." },
  { key: "metric", label: "Metric", type: "select", options: ["position", "displacement"] },
  { key: "time", label: "Frame count mismatch", type: "select", options: ["truncate", "resample"] },
  { key: "radius", label: "Dot radius (px)", type: "number", min: 0 },
  { key: "vmax", label: "Color max (px)", type: "number", min: 0, help: "diff view only; empty = 95th percentile." },
];
const COMPARE_DEFAULTS = { view: "pairs", metric: "position", time: "truncate", radius: 2, vmax: "" };

// ------------------------------------------------------------------ state

const ROW_NAMES = ["Row 1", "Row 2"];
const COL_NAMES = ["A", "B", "Compare"];

const S = {
  blocks: [],
  frame: 0,
  maxFrames: 0,
  masterFps: 30,
  playing: false,
  speed: 1,
  t0: 0,
  f0: 0,
  gpus: [],
};

// A block shows either its source video or its result video (tracked / compared).
function shown(b) {
  if (b.view === "result" && b.result) return b.result.meta;
  return b.src;
}

// ------------------------------------------------------------------ block UI

function makeBlock(row, col) {
  const isCompare = col === 2;
  const b = {
    id: `r${row + 1}${"abc"[col]}`, row, col, isCompare,
    src: null, result: null, view: "source", job: null, poll: null,
    params: structuredClone(isCompare ? COMPARE_DEFAULTS : TRACK_DEFAULTS),
  };
  const el = document.createElement("section");
  el.className = "block";
  el.innerHTML = `
    <div class="bhead">
      <span class="title">${ROW_NAMES[row]} · ${COL_NAMES[col]}</span>
      <span class="badge"></span>
      <span class="name"></span>
      <div class="actions">
        ${isCompare ? "" : '<button data-act="select">Select video</button>'}
        <button data-act="params" title="Parameters">⚙</button>
        <button data-act="run" class="primary">${isCompare ? "Compare A vs B" : "Track"}</button>
        <button data-act="cancel" hidden>Cancel</button>
        ${isCompare ? "" : '<button data-act="toggle" hidden>Show source</button>'}
        <button data-act="clear" class="danger">Clear</button>
      </div>
    </div>
    <div class="stage">
      <video muted playsinline preload="auto"></video>
      <div class="placeholder"><div>${isCompare
        ? "Track A and B in this row, then press <b>Compare A vs B</b>.<br>Dots are drawn on video A."
        : "Press <b>Select video</b> to choose a source video."}</div></div>
      <pre class="status"></pre>
    </div>`;
  b.el = el;
  b.video = $("video", el);
  b.video.addEventListener("loadedmetadata", () => seekBlock(b, S.frame));
  el.addEventListener("click", (e) => {
    const act = e.target.closest("button")?.dataset.act;
    if (act) onAction(b, act);
  });
  return b;
}

function render(b) {
  const meta = shown(b);
  const running = b.job && b.job.status === "running";
  const badge = $(".badge", b.el);
  if (b.result && b.view === "result") {
    badge.textContent = b.isCompare ? "compared" : "tracked";
    badge.className = "badge " + (b.isCompare ? "compared" : "tracked");
  } else {
    badge.textContent = running ? "running…" : b.src ? "source" : "empty";
    badge.className = "badge";
  }
  let name = meta ? basename(meta.path) : "";
  if (b.result && b.view === "result") {
    name = b.isCompare ? `${b.result.names[0]} vs ${b.result.names[1]}` : `${basename(b.src.path)} (tracked)`;
  }
  $(".name", b.el).textContent = name;
  $(".name", b.el).title = meta ? meta.path : "";
  $(".placeholder", b.el).hidden = !!meta;
  b.video.hidden = !meta;

  const run = $('[data-act="run"]', b.el);
  run.disabled = running || (!b.isCompare && !b.src);
  run.textContent = b.isCompare ? "Compare A vs B" : b.result ? "Re-track" : "Track";
  $('[data-act="cancel"]', b.el).hidden = !running;
  const toggle = $('[data-act="toggle"]', b.el);
  if (toggle) {
    toggle.hidden = !(b.src && b.result);
    toggle.textContent = b.view === "result" ? "Show source" : "Show tracked";
  }
  const sel = $('[data-act="select"]', b.el);
  if (sel) sel.disabled = running;
}

function setStatus(b, text, failed = false) {
  const st = $(".status", b.el);
  st.textContent = text || "";
  st.classList.toggle("failed", failed);
  st.scrollTop = st.scrollHeight;
}

function showVideo(b) {
  const meta = shown(b);
  if (!meta) {
    b.video.removeAttribute("src");
    b.video.load();
  } else if (b.video.dataset.url !== meta.url) {
    b.video.dataset.url = meta.url;
    b.video.src = meta.url;
  }
  if (!meta) delete b.video.dataset.url;
  render(b);
  refreshTimeline();
}

async function onAction(b, act) {
  try {
    if (act === "select") openPicker(b);
    else if (act === "params") openParams(b, false);
    else if (act === "run") openParams(b, true);
    else if (act === "cancel") await cancelJob(b);
    else if (act === "clear") await clearBlock(b);
    else if (act === "toggle") {
      b.view = b.view === "result" ? "source" : "result";
      showVideo(b);
    }
  } catch (err) {
    toast(err.message);
  }
}

async function setSource(b, path) {
  setStatus(b, "Loading video info…");
  const meta = await api(`/api/meta?path=${encodeURIComponent(path)}`);
  stopPolling(b);
  b.src = meta;
  b.result = null;
  b.job = null;
  b.view = "source";
  setStatus(b, "");
  showVideo(b);
}

async function clearBlock(b) {
  if (b.job && b.job.status === "running") await cancelJob(b);
  stopPolling(b);
  b.src = null;
  b.result = null;
  b.job = null;
  b.view = "source";
  setStatus(b, "");
  showVideo(b);
}

// ------------------------------------------------------------------ jobs

function rowBlock(row, col) {
  return S.blocks.find((x) => x.row === row && x.col === col);
}

async function runJob(b) {
  let job;
  if (b.isCompare) {
    const A = rowBlock(b.row, 0), B = rowBlock(b.row, 1);
    if (!A.result || !B.result) {
      toast(`Track both A and B in ${ROW_NAMES[b.row]} first.`);
      return;
    }
    const p = b.params;
    b.names = [A, B].map((x) => basename(x.src.path));
    job = await postJSON("/api/compare", {
      slot: b.id, a: A.result.dir, b: B.result.dir, view: p.view, metric: p.metric,
      time: p.time, radius: Number(p.radius), vmax: p.vmax === "" ? null : Number(p.vmax),
    });
  } else {
    const p = b.params;
    if (!p.gpus.length && S.gpus.length) {
      toast("Select at least one GPU.");
      return;
    }
    job = await postJSON("/api/track", {
      slot: b.id, video: b.src.path, gpus: p.gpus, mode: p.mode, grid_size: String(p.grid_size),
      radius: Number(p.radius), chunk_size: Number(p.chunk_size), frame_stride: Number(p.frame_stride),
      max_frames: Number(p.max_frames), grid_query_frame: Number(p.grid_query_frame),
      backward_tracking: !!p.backward_tracking, mask: p.mask || "",
    });
  }
  b.job = { id: job.job, status: "running" };
  setStatus(b, "Starting…");
  render(b);
  stopPolling(b);
  b.poll = setInterval(() => pollJob(b), 1000);
}

async function pollJob(b) {
  if (!b.job) return stopPolling(b);
  let info;
  try {
    info = await api(`/api/jobs/${b.job.id}`);
  } catch (err) {
    stopPolling(b);
    setStatus(b, `Lost job: ${err.message}`, true);
    return;
  }
  b.job.status = info.status;
  if (info.status === "running") {
    setStatus(b, `Running ${info.elapsed}s…\n${info.tail}`);
  } else {
    stopPolling(b);
    if (info.status === "done") {
      const meta = await api(`/api/meta?path=${encodeURIComponent(info.result_video)}`);
      b.result = { dir: info.result_dir, meta, names: b.names || [] };
      b.view = "result";
      setStatus(b, "");
      showVideo(b);
      toast(`${ROW_NAMES[b.row]} · ${COL_NAMES[b.col]}: ${b.isCompare ? "compare" : "tracking"} finished`);
    } else {
      setStatus(b, `${info.status.toUpperCase()} (exit ${info.returncode ?? "-"})\n${info.tail}`, true);
    }
  }
  render(b);
}

function stopPolling(b) {
  clearInterval(b.poll);
  b.poll = null;
}

async function cancelJob(b) {
  if (!b.job) return;
  await postJSON(`/api/jobs/${b.job.id}/cancel`, {});
  b.job.status = "cancelled";
  stopPolling(b);
  setStatus(b, "Cancelled.", true);
  render(b);
}

// ------------------------------------------------------------------ picker dialog

let pickerTarget = null;
let pickerData = { dir: "", parent: null, dirs: [], files: [] };
const LAST_DIR_KEY = "cotracker-gui.lastDir";

// The last folder a video was picked from, kept in this browser (survives reloads).
function lastDir() {
  try { return localStorage.getItem(LAST_DIR_KEY) || "source_video"; } catch { return "source_video"; }
}
function rememberDir(dir) {
  try { localStorage.setItem(LAST_DIR_KEY, dir); } catch { /* storage unavailable */ }
}
const dirOf = (path) => (path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : ".");

async function openPicker(b) {
  pickerTarget = b;
  $("#picker-status").textContent = "";
  $("#picker").showModal();
  try {
    await browseTo(lastDir());
  } catch {
    await browseTo("");  // remembered folder is gone: show the start locations
  }
  $("#picker-filter").focus();
}

async function browseTo(dir) {
  $("#picker-list").innerHTML = '<li class="empty">Loading…</li>';
  pickerData = await api(`/api/browse?kind=video&dir=${encodeURIComponent(dir)}`);
  $("#picker-filter").value = "";
  drawPickerList();
}

function drawPickerList() {
  const q = $("#picker-filter").value.toLowerCase();
  const list = $("#picker-list");
  const d = pickerData;
  $("#picker-path").textContent = d.dir === "" ? "Start locations" : d.dir === "." ? "zeyu-co-tracker/" : `${d.dir}/`;
  list.innerHTML = "";
  const add = (text, cls, onclick) => {
    const li = document.createElement("li");
    li.textContent = text;
    li.className = cls;
    li.onclick = onclick;
    list.appendChild(li);
  };
  if (d.parent !== null) add("⬑ ..", "dir", () => browseTo(d.parent).catch((e) => toast(e.message)));
  const match = (p) => basename(p).toLowerCase().includes(q);
  for (const sub of d.dirs.filter(match)) {
    const label = d.dir === "" ? (sub === "." ? "zeyu-co-tracker (repo root)" : sub) : basename(sub);
    add(`📁 ${label}/`, "dir", () => browseTo(sub).catch((e) => toast(e.message)));
  }
  for (const f of d.files.filter(match)) add(`🎞 ${basename(f)}`, "file", () => choose(f));
  if (!list.children.length || (d.parent !== null && list.children.length === 1)) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = q ? "Nothing matches the filter." : "No videos or folders here.";
    list.appendChild(li);
  }
}

async function choose(path, remember = true) {
  if (remember) rememberDir(dirOf(path));
  $("#picker").close();
  try {
    await setSource(pickerTarget, path);
  } catch (err) {
    setStatus(pickerTarget, err.message, true);
  }
}

$("#picker-filter").addEventListener("input", drawPickerList);
// Upload from the browser's computer. Browsers never reveal a file's folder, so the server
// cannot open it in place; it receives a copy (cached in outputs/gui/.uploads). Chrome/Edge's
// showOpenFilePicker remembers the last local folder per `id`, shared by all blocks.
async function uploadFile(file) {
  $("#picker-status").textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const { path } = await api("/api/upload", { method: "POST", body: fd });
    await choose(path, false);
  } catch (err) {
    $("#picker-status").textContent = err.message;
  }
}

$("#picker-upload-btn").addEventListener("click", async () => {
  if (!window.showOpenFilePicker) return $("#picker-upload").click();
  let handle;
  try {
    [handle] = await window.showOpenFilePicker({
      id: "cotracker-gui-video",
      types: [{ description: "Videos", accept: { "video/*": [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"] } }],
    });
  } catch (err) {
    if (err.name !== "AbortError") $("#picker-status").textContent = err.message;
    return;
  }
  await uploadFile(await handle.getFile());
});
$("#picker-upload").addEventListener("change", async (e) => {
  if (e.target.files[0]) await uploadFile(e.target.files[0]);
  e.target.value = "";
});

// ------------------------------------------------------------------ params dialog

let paramsTarget = null;

async function openParams(b, runAfter) {
  if (!b.isCompare && !b.src && runAfter) return toast("Select a source video first.");
  paramsTarget = b;
  const fields = b.isCompare ? COMPARE_FIELDS : TRACK_FIELDS;
  $("#params-title").textContent =
    `${ROW_NAMES[b.row]} · ${COL_NAMES[b.col]} — ${b.isCompare ? "compare" : "tracking"} parameters`;
  $("#params-run").textContent = b.isCompare ? "Save & compare" : "Save & track";
  const box = $("#params-fields");
  box.innerHTML = "";
  if (!b.isCompare) {
    try { S.gpus = (await api("/api/gpus")).gpus; } catch { /* keep last list */ }
  }
  for (const f of fields) {
    const lab = document.createElement("label");
    lab.className = "key";
    lab.textContent = f.label;
    box.appendChild(lab);
    box.appendChild(await fieldInput(f, b.params[f.key]));
    if (f.help) {
      const h = document.createElement("div");
      h.className = "help";
      h.textContent = f.help;
      box.appendChild(h);
    }
  }
  $("#params").showModal();
}

async function fieldInput(f, value) {
  if (f.type === "select") {
    const s = document.createElement("select");
    s.name = f.key;
    for (const o of f.options) s.add(new Option(o, o, false, o === value));
    return s;
  }
  if (f.type === "checkbox") {
    const c = document.createElement("input");
    c.type = "checkbox";
    c.name = f.key;
    c.checked = !!value;
    return c;
  }
  if (f.type === "gpus") {
    const d = document.createElement("div");
    d.className = "gpu-list";
    if (!S.gpus.length) d.textContent = "No GPU found (runs on CPU).";
    for (const g of S.gpus) {
      const l = document.createElement("label");
      const c = document.createElement("input");
      c.type = "checkbox";
      c.name = "gpus";
      c.value = g.index;
      c.checked = value.includes(g.index);
      const used = (g.used_mb / 1024).toFixed(1), tot = (g.total_mb / 1024).toFixed(0);
      l.append(c, ` GPU ${g.index}: ${g.name} — ${used} / ${tot} GB in use`);
      d.appendChild(l);
    }
    return d;
  }
  if (f.type === "mask") {
    const s = document.createElement("select");
    s.name = f.key;
    s.add(new Option("(none)", ""));
    try {
      for (const m of (await api("/api/files?kind=image")).files) s.add(new Option(m, m, false, m === value));
    } catch { /* leave only (none) */ }
    return s;
  }
  const i = document.createElement("input");
  i.type = f.type;
  i.name = f.key;
  if (f.min !== undefined) i.min = f.min;
  i.value = value ?? "";
  return i;
}

function readParams(b) {
  const box = $("#params-fields");
  const fields = b.isCompare ? COMPARE_FIELDS : TRACK_FIELDS;
  for (const f of fields) {
    if (f.type === "gpus") {
      b.params.gpus = [...box.querySelectorAll('input[name="gpus"]:checked')].map((c) => Number(c.value));
    } else if (f.type === "checkbox") {
      b.params[f.key] = box.querySelector(`[name="${f.key}"]`).checked;
    } else {
      b.params[f.key] = box.querySelector(`[name="${f.key}"]`).value;
    }
  }
}

$("#params").addEventListener("close", async () => {
  const b = paramsTarget;
  if (!b) return;
  readParams(b);
  if ($("#params").returnValue === "run") {
    try {
      await runJob(b);
    } catch (err) {
      setStatus(b, err.message, true);
      render(b);
    }
  }
});

// ------------------------------------------------------------------ synced playback
// Like zeyu-video_viewer: all blocks show the same frame index. The master clock runs at the
// highest fps among loaded videos; each video's playbackRate is scaled so its frames advance
// in step, and drift is corrected by seeking.

function loaded() {
  return S.blocks.filter((b) => shown(b));
}

function refreshTimeline() {
  const ls = loaded();
  S.maxFrames = Math.max(0, ...ls.map((b) => shown(b).frames));
  S.masterFps = Math.max(1, ...ls.map((b) => shown(b).fps));
  const slider = $("#slider");
  slider.max = Math.max(0, S.maxFrames - 1);
  if (S.frame > slider.max) S.frame = Number(slider.max);
  if (!ls.length) pause();
  updateLabel();
  if (!S.playing) seekAll(S.frame);
}

function updateLabel() {
  $("#slider").value = S.frame;
  $("#frame-label").textContent = `Frame ${S.frame} / ${Math.max(0, S.maxFrames - 1)}`;
}

function seekBlock(b, frame) {
  const m = shown(b);
  if (!m) return;
  const ended = frame >= m.frames;
  b.video.classList.toggle("ended", ended);
  if (ended) return;
  const t = (frame + 0.5) / m.fps;
  if (Math.abs(b.video.currentTime - t) > 0.25 / m.fps) b.video.currentTime = t;
}

function seekAll(frame) {
  for (const b of loaded()) seekBlock(b, frame);
}

function play() {
  if (!loaded().length) return;
  if (S.frame >= S.maxFrames - 1) S.frame = 0;
  S.playing = true;
  S.t0 = performance.now();
  S.f0 = S.frame;
  for (const b of loaded()) {
    const m = shown(b);
    b.video.playbackRate = (S.speed * S.masterFps) / m.fps;
    seekBlock(b, S.frame);
    if (S.frame < m.frames) b.video.play().catch(() => {});
  }
  $("#btn-play").textContent = "Pause";
  requestAnimationFrame(tick);
}

function pause() {
  S.playing = false;
  for (const b of S.blocks) b.video.pause();
  $("#btn-play").textContent = "Play";
  seekAll(S.frame);
}

function tick(now) {
  if (!S.playing) return;
  const fl = S.f0 + ((now - S.t0) / 1000) * S.masterFps * S.speed; // fractional master frame
  if (fl >= S.maxFrames - 1) {
    S.frame = Math.max(0, S.maxFrames - 1);
    pause();
    updateLabel();
    return;
  }
  S.frame = Math.floor(fl);
  for (const b of loaded()) {
    const m = shown(b);
    const v = b.video;
    if (S.frame >= m.frames) {
      if (!v.paused) v.pause();
      v.classList.add("ended");
      continue;
    }
    v.classList.remove("ended");
    const target = (fl + 0.5) / m.fps;
    v.playbackRate = (S.speed * S.masterFps) / m.fps;
    if (Math.abs(v.currentTime - target) > 2 / m.fps) v.currentTime = target;
    if (v.paused) v.play().catch(() => {});
  }
  updateLabel();
  requestAnimationFrame(tick);
}

function step(delta) {
  pause();
  S.frame = Math.min(Math.max(0, S.frame + delta), Math.max(0, S.maxFrames - 1));
  updateLabel();
  seekAll(S.frame);
}

$("#btn-play").onclick = () => (S.playing ? pause() : play());
$("#btn-prev").onclick = () => step(-1);
$("#btn-next").onclick = () => step(1);
$("#btn-first").onclick = () => step(-S.frame);
$("#slider").addEventListener("input", (e) => {
  if (S.playing) pause();
  S.frame = Number(e.target.value);
  updateLabel();
  seekAll(S.frame);
});
$("#speed").addEventListener("change", (e) => {
  const wasPlaying = S.playing;
  if (wasPlaying) pause();
  S.speed = Number(e.target.value);
  if (wasPlaying) play();
});
document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea, dialog")) return;
  if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
  else if (e.code === "ArrowLeft") step(-1);
  else if (e.code === "ArrowRight") step(1);
});

// ------------------------------------------------------------------ init

for (let row = 0; row < 2; row++) {
  for (let col = 0; col < 3; col++) {
    const b = makeBlock(row, col);
    S.blocks.push(b);
    $("#grid").appendChild(b.el);
    render(b);
  }
}
api("/api/gpus").then((r) => {
  S.gpus = r.gpus;
  // Default to the GPU with the most free memory.
  if (S.gpus.length) {
    const best = S.gpus.reduce((a, g) => (g.total_mb - g.used_mb > a.total_mb - a.used_mb ? g : a));
    for (const b of S.blocks) if (!b.isCompare) b.params.gpus = [best.index];
  }
}).catch(() => {});
refreshTimeline();
