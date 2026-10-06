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
// Name of the video behind a track block (source file, or the loaded result folder).
const videoName = (b) => (b.src ? basename(b.src.path) : b.result ? b.result.label : "");

// ------------------------------------------------------------------ parameters

const TRACK_FIELDS = [
  { key: "gpus", label: "GPUs", type: "gpus", help: "Points are split across the selected GPUs." },
  { key: "mode", label: "Mode", type: "select", options: ["offline", "online"],
    help: "offline: whole clip at once (most accurate). online: sliding window, for long videos." },
  { key: "grid_size", label: "Grid size", type: "text", help: '"30" = 30×30 points, or COLSxROWS like "100x50".' },
  { key: "radius", label: "Dot radius (px)", type: "number", min: 0 },
  { key: "chunk_size", label: "Chunk size", type: "number", min: 0,
    help: "Max points per forward pass; lower it on out-of-memory. 0 = split evenly over GPUs." },
  { key: "segment_len", label: "Segment length", type: "number", min: 0,
    help: "Videos longer than this restart tracking every N frames (fresh grid, no overlap). 0 = whole video at once." },
  { key: "frame_stride", label: "Frame stride", type: "number", min: 1, help: "Use every k-th frame." },
  { key: "max_frames", label: "Max frames", type: "number", min: 0, help: "0 = all frames." },
  { key: "grid_query_frame", label: "Grid start frame", type: "number", min: 0,
    help: "Only used when the whole video is tracked at once (no segments)." },
  { key: "backward_tracking", label: "Backward tracking", type: "checkbox", help: "Offline only." },
  { key: "mask", label: "Mask image", type: "mask", help: "Optional: keep only grid points inside the mask." },
];
const TRACK_DEFAULTS = {
  gpus: [0], mode: "offline", grid_size: "200x100", radius: 2, chunk_size: 0, frame_stride: 1,
  max_frames: 0, grid_query_frame: 0, segment_len: 10, backward_tracking: false, mask: "",
};

const COMPARE_FIELDS = [
  { key: "direction", label: "Direction", type: "select", options: ["A vs B", "B vs A"],
    help: "First video is the reference: dots are drawn on it, and the difference is first − second." },
  { key: "view", label: "View", type: "select", options: ["pairs", "diff"],
    help: "pairs: green = point has a visible pair in B, gray = no pair. diff: color by |A − B|." },
  { key: "metric", label: "Metric", type: "select", options: ["position", "displacement"] },
  { key: "time", label: "Frame count mismatch", type: "select", options: ["truncate", "resample"] },
  { key: "radius", label: "Dot radius (px)", type: "number", min: 0 },
  { key: "vmax", label: "Color max (px)", type: "number", min: 0, help: "diff view only; empty = 95th percentile." },
  { key: "mask", label: "Mask (.npy)", type: "npypath",
    help: "Optional. (T, H, W) bool, same size and length as the sources (e.g. *_mask_stack.npy); only points inside its True area are compared (video, numbers and curve)." },
  { key: "mask_step", label: "Mask step (frames)", type: "number", min: 1,
    help: "Consult the mask every N frames (0, N, 2N, …); the selected points hold until the next one." },
  { key: "mask_rule", label: "Point counts if inside", type: "select",
    options: [["both", "the mask in both A and B"], ["a", "the mask in the first video"], ["either", "the mask in either video"]] },
];
const COMPARE_DEFAULTS = {
  direction: "A vs B", view: "pairs", metric: "position", time: "truncate", radius: 2, vmax: "",
  mask: "", mask_step: 5, mask_rule: "both",
};

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
        <button data-act="select" title="${isCompare ? "Load an existing video, e.g. an earlier compare.mp4" : "Choose the source video"}">Select video</button>
        <button data-act="params" title="Parameters">⚙</button>
        <button data-act="run" class="primary">${isCompare ? "Compare A vs B" : "Track"}</button>
        <button data-act="cancel" hidden>Cancel</button>
        <button data-act="toggle" hidden>Show source</button>
        ${isCompare ? '<button data-act="curve" hidden title="Per-frame difference between the two videos">Curve</button>' : ""}
        <button data-act="clear" class="danger">Clear</button>
      </div>
    </div>
    <div class="stage">
      <video muted playsinline preload="auto"></video>
      ${isCompare ? `<canvas class="curve" hidden></canvas>
      <label class="ymax" hidden title="Upper bound of the curve's y axis; empty = automatic">y max
        <input type="number" min="0" step="any" placeholder="auto"> px</label>` : ""}
      <div class="placeholder"><div>${isCompare
        ? "Track A and B in this row, then press <b>Compare</b> (⚙ for A vs B or B vs A),<br>or <b>Select video</b> to load an existing one."
        : "Press <b>Select video</b> to choose a source video."}</div></div>
      <pre class="status"></pre>
    </div>`;
  b.el = el;
  b.video = $("video", el);
  b.video.addEventListener("loadedmetadata", () => seekBlock(b, S.frame));
  if (isCompare) setupCurve(b);
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
    badge.textContent = running ? "running…" : b.src ? (b.isCompare ? "loaded" : "source") : "empty";
    badge.className = "badge";
  }
  let name = meta ? basename(meta.path) : "";
  if (b.result && b.view === "result") {
    name = b.isCompare ? `${b.result.names[0]} vs ${b.result.names[1]}`
      : b.result.loaded ? `${b.result.label} (loaded result)` : `${videoName(b)} (tracked)`;
  }
  $(".name", b.el).textContent = name;
  $(".name", b.el).title = meta ? meta.path : "";
  $(".placeholder", b.el).hidden = !!meta;
  b.video.hidden = !meta;

  const run = $('[data-act="run"]', b.el);
  run.disabled = running || (!b.isCompare && !b.src);
  run.textContent = b.isCompare ? `Compare ${b.params.direction}` : b.result ? "Re-track" : "Track";
  $('[data-act="cancel"]', b.el).hidden = !running;
  const toggle = $('[data-act="toggle"]', b.el);
  if (toggle) {
    toggle.hidden = !(b.src && b.result);
    toggle.textContent = b.isCompare
      ? (b.view === "result" ? "Show loaded" : "Show compared")
      : (b.view === "result" ? "Show source" : "Show tracked");
  }
  const sel = $('[data-act="select"]', b.el);
  if (sel) sel.disabled = running;
  if (b.isCompare) {
    const stats = meta && meta.stats;
    const curveBtn = $('[data-act="curve"]', b.el);
    curveBtn.hidden = !stats;
    curveBtn.textContent = b.showCurve ? "Hide curve" : "Curve";
    const on = !!(stats && b.showCurve);
    $(".stage", b.el).classList.toggle("with-curve", on);
    $("canvas.curve", b.el).hidden = !on;
    $("label.ymax", b.el).hidden = !on;
    if (on) drawCurve(b);
  }
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
  if (b.isCompare && meta && meta.stats === undefined) {
    loadStats(meta).then(() => render(b));
  }
}

async function onAction(b, act) {
  try {
    if (act === "select") openPicker(b);
    else if (act === "params") openParams(b, false);
    else if (act === "run") openParams(b, true);
    else if (act === "cancel") await cancelJob(b);
    else if (act === "clear") await clearBlock(b);
    else if (act === "curve") {
      b.showCurve = !b.showCurve;
      render(b);
    } else if (act === "toggle") {
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
      toast(`Track or load tracking results for both A and B in ${ROW_NAMES[b.row]} first.`);
      return;
    }
    const p = b.params;
    const [first, second] = p.direction === "B vs A" ? [B, A] : [A, B];  // first = reference, drawn on
    b.names = [first, second].map(videoName);
    if (first.result.loaded && !first.result.hasMeta && !first.src) {
      toast(`${COL_NAMES[first.col]} (${first.result.label}) has no meta.json, so its source video is unknown. ` +
            "It can only be the second video: switch Direction in ⚙.");
      return;
    }
    job = await postJSON("/api/compare", {
      slot: b.id, a: first.result.dir, b: second.result.dir, video_a: first.src ? first.src.path : "",
      view: p.view, metric: p.metric,
      time: p.time, radius: Number(p.radius), vmax: p.vmax === "" ? null : Number(p.vmax),
      mask: (p.mask || "").trim(), mask_step: Number(p.mask_step) || 5, mask_rule: p.mask_rule,
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
      segment_len: Number(p.segment_len),
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

let pickerCallback = null;  // set when the picker fills a form field instead of loading a block
let pickerKind = "video";   // which files the picker lists: "video" or "npy"

async function openPicker(b, onPick = null, kind = "video") {
  pickerTarget = b;
  pickerCallback = onPick;
  pickerKind = kind;
  $("#picker h2").textContent =
    kind === "npy" ? "Select mask (.npy)" : b.isCompare ? "Load a video" : "Select source video";
  $("#picker-upload-btn").hidden = kind !== "video";
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
  pickerData = await api(`/api/browse?kind=${pickerKind}&dir=${encodeURIComponent(dir)}`);
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
  const results = new Set(d.results || []);
  for (const sub of d.dirs.filter(match)) {
    const label = d.dir === "" ? (sub === "." ? "zeyu-co-tracker (repo root)" : sub) : basename(sub);
    if (results.has(sub) && !pickerTarget.isCompare && !pickerCallback) {
      add(`📊 ${label}/  — tracking result, click to load`, "dir result", () => chooseResult(sub));
    } else {
      add(`📁 ${label}/`, "dir", () => browseTo(sub).catch((e) => toast(e.message)));
    }
  }
  const icon = pickerKind === "npy" ? "🧩" : "🎞";
  for (const f of d.files.filter(match)) add(`${icon} ${basename(f)}`, "file", () => choose(f));
  if (!list.children.length || (d.parent !== null && list.children.length === 1)) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = q ? "Nothing matches the filter." : "No videos or folders here.";
    list.appendChild(li);
  }
}

// Load an existing tracking result folder (tracks.npy, tracks.mp4, meta.json) into an A/B block.
async function chooseResult(dir) {
  rememberDir(dirOf(dir));
  $("#picker").close();
  const b = pickerTarget;
  try {
    setStatus(b, "Loading tracking result…");
    const info = await api(`/api/result?dir=${encodeURIComponent(dir)}`);
    const metaOf = (p) => (p ? api(`/api/meta?path=${encodeURIComponent(p)}`) : null);
    const [srcMeta, resMeta] = await Promise.all([metaOf(info.source), metaOf(info.video)]);
    if (b.job && b.job.status === "running") await cancelJob(b);
    stopPolling(b);
    b.job = null;
    b.src = srcMeta;
    b.result = { dir: info.dir, meta: resMeta, loaded: true, hasMeta: info.has_meta, label: basename(info.dir) };
    b.view = resMeta ? "result" : "source";
    const notes = [];
    if (!info.video) notes.push("no tracks.mp4 in this folder (it can still be compared)");
    if (!info.has_meta) notes.push("no meta.json: source video unknown, use it as the second video in Compare");
    else if (!info.source) notes.push("its source video (from meta.json) was not found");
    setStatus(b, notes.length ? "Note: " + notes.join("; ") + "." : "");
    showVideo(b);
  } catch (err) {
    setStatus(b, err.message, true);
  }
}

async function choose(path, remember = true) {
  if (remember) rememberDir(dirOf(path));
  $("#picker").close();
  if (pickerCallback) {
    pickerCallback(path);
    pickerCallback = null;
    return;
  }
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
    for (const o of f.options) {
      const [v, label] = Array.isArray(o) ? o : [o, o];
      s.add(new Option(label, v, false, v === value));
    }
    return s;
  }
  if (f.type === "npypath") {
    const d = document.createElement("div");
    d.className = "pathfield";
    const i = document.createElement("input");
    i.type = "text";
    i.name = f.key;
    i.value = value || "";
    i.placeholder = "(none)";
    const browse = document.createElement("button");
    browse.type = "button";
    browse.textContent = "Browse…";
    browse.onclick = () => openPicker(paramsTarget, (path) => { i.value = path; }, "npy");
    const clear = document.createElement("button");
    clear.type = "button";
    clear.textContent = "×";
    clear.title = "No mask";
    clear.onclick = () => { i.value = ""; };
    d.append(i, browse, clear);
    return d;
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
  render(b);
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
  for (const b of S.blocks) if (b.isCompare && b.showCurve) drawCurve(b);
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

// ------------------------------------------------------------------ difference curve
// Per-frame mean / median difference (per_frame.csv of the compare run),
// with a marker at the current frame; hover shows values, click jumps to that frame.

const CURVE_SERIES = [
  { key: "mean", label: "mean", light: "#2a78d6", dark: "#3987e5", dash: [] },
  { key: "median", label: "median", light: "#eb6834", dark: "#d95926", dash: [] },
];
const darkMode = window.matchMedia("(prefers-color-scheme: dark)");

async function loadStats(meta) {
  try {
    meta.stats = await api(`/api/compare_stats?path=${encodeURIComponent(meta.path)}`);
  } catch {
    meta.stats = null; // not a compare video (no per_frame.csv next to it)
  }
}

function setupCurve(b) {
  const cv = $("canvas.curve", b.el);
  $("label.ymax input", b.el).addEventListener("input", (e) => {
    const v = parseFloat(e.target.value);
    b.curveYMax = v > 0 ? v : null;  // empty / 0 -> automatic
    drawCurve(b);
  });
  cv.addEventListener("mousemove", (e) => {
    b.hoverFrame = curveFrameAt(b, e);
    drawCurve(b);
  });
  cv.addEventListener("mouseleave", () => {
    b.hoverFrame = null;
    drawCurve(b);
  });
  cv.addEventListener("click", (e) => {
    const f = curveFrameAt(b, e);
    if (f == null) return;
    if (S.playing) pause();
    S.frame = Math.min(f, Math.max(0, S.maxFrames - 1));
    updateLabel();
    seekAll(S.frame);
  });
  new ResizeObserver(() => b.showCurve && drawCurve(b)).observe($(".stage", b.el));
  darkMode.addEventListener("change", () => b.showCurve && drawCurve(b));
}

function curveGeom(b) {
  const cv = $("canvas.curve", b.el);
  const w = cv.clientWidth, h = cv.clientHeight;
  return { cv, w, h, left: 44, right: 64, top: 26, bottom: 38 };  // bottom: x labels + summary line
}

function curveFrameAt(b, e) {
  const st = shown(b)?.stats;
  if (!st) return null;
  const g = curveGeom(b);
  const n = st.frame.length;
  const x = e.offsetX - g.left, pw = g.w - g.left - g.right;
  if (x < -4 || x > pw + 4 || n < 1) return null;
  return Math.max(0, Math.min(n - 1, Math.round((x / Math.max(1, pw)) * (n - 1))));
}

// Round tick step (1, 2 or 5 × 10^k) giving about `ticks` intervals up to v.
function niceStep(v, ticks = 4) {
  if (!(v > 0)) return 1;
  const raw = v / ticks, p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}

function drawCurve(b) {
  const meta = shown(b);
  const st = meta && meta.stats;
  const g = curveGeom(b);
  const { cv, w, h } = g;
  if (!st || !w || !h || cv.hidden) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  const ink = css.getPropertyValue("--text").trim();
  const muted = css.getPropertyValue("--muted").trim();
  const grid = css.getPropertyValue("--border").trim();
  const surface = css.getPropertyValue("--panel").trim();
  const mode = darkMode.matches ? "dark" : "light";
  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, w, h);

  const n = st.frame.length;
  const pw = w - g.left - g.right, ph = h - g.top - g.bottom;
  if (n < 1 || pw < 20 || ph < 20) return;
  let ymax, step, nTicks;
  if (b.curveYMax) {  // user-set upper bound; larger values are clipped at the top
    ymax = b.curveYMax;
    step = niceStep(ymax);
    nTicks = Math.max(1, Math.floor(ymax / step + 1e-9));
  } else {
    const top = Math.max(0, ...CURVE_SERIES.flatMap((s) => st[s.key].filter((v) => v != null)));
    step = niceStep(top);
    nTicks = Math.max(1, Math.ceil(top / step));
    ymax = step * nTicks;
  }
  const X = (i) => g.left + (n === 1 ? pw / 2 : (i / (n - 1)) * pw);
  const Y = (v) => g.top + ph - (v / ymax) * ph;
  ctx.font = "11px system-ui, sans-serif";

  // recessive grid + y labels
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let k = 0; k <= nTicks; k++) {
    const v = step * k, y = Math.round(Y(v)) + 0.5;
    ctx.strokeStyle = grid;
    ctx.beginPath(); ctx.moveTo(g.left, y); ctx.lineTo(g.left + pw, y); ctx.stroke();
    ctx.fillStyle = muted;
    ctx.fillText(`${+v.toFixed(3)}`, g.left - 6, y);
  }
  // x labels: first / last frame
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(`${st.frame[0]}`, g.left, g.top + ph + 5);
  ctx.textAlign = "right";
  ctx.fillText(`frame ${st.frame[n - 1]}`, g.left + pw, g.top + ph + 5);
  // whole-run summary (summary.json of the compare run): area under each curve, overall mean/median
  const sm = st.summary || {};
  const num = (v, d = 2) => (v == null ? "–" : (+v).toFixed(d));
  const summaryText = `AUC mean ${num(sm.auc_mean, 1)} · median ${num(sm.auc_median, 1)} px·fr` +
    `   overall mean ${num(sm.overall_mean)} · median ${num(sm.overall_median)} px`;
  ctx.fillStyle = ink;
  ctx.textAlign = "center";
  let fs = 11;  // shrink to fit narrow blocks
  while (fs > 8 && ctx.measureText(summaryText).width > w - 8) ctx.font = `${--fs}px system-ui, sans-serif`;
  ctx.fillText(summaryText, w / 2, g.top + ph + 21);
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = muted;
  ctx.save();
  ctx.translate(11, g.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("px", 0, 0);
  ctx.restore();

  // series lines (gaps where a frame has no valid points), clipped to the plot area
  ctx.save();
  ctx.beginPath();
  ctx.rect(g.left - 5, g.top, pw + 10, ph + 5);
  ctx.clip();
  for (const s of CURVE_SERIES) {
    ctx.strokeStyle = s[mode];
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.setLineDash(s.dash);
    ctx.beginPath();
    let pen = false;
    st[s.key].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      pen ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v));
      pen = true;
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
  // direct labels at line ends (text in ink, short color key beside it), nudged apart
  const ends = CURVE_SERIES.map((s) => {
    let i = n - 1;
    while (i > 0 && st[s.key][i] == null) i--;
    return { s, y: st[s.key][i] == null ? null : Math.max(g.top, Y(st[s.key][i])) };
  }).filter((e) => e.y != null).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 13);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const e of ends) {
    ctx.strokeStyle = e.s[mode];
    ctx.lineWidth = 2;
    ctx.setLineDash(e.s.dash.length ? [3, 2] : []);
    ctx.beginPath(); ctx.moveTo(g.left + pw + 4, e.y); ctx.lineTo(g.left + pw + 12, e.y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ink;
    ctx.fillText(e.s.label, g.left + pw + 15, e.y);
  }
  // legend (top-left)
  let lx = g.left;
  ctx.textBaseline = "middle";
  for (const s of CURVE_SERIES) {
    ctx.strokeStyle = s[mode];
    ctx.lineWidth = 2;
    ctx.setLineDash(s.dash.length ? [3, 2] : []);
    ctx.beginPath(); ctx.moveTo(lx, 11); ctx.lineTo(lx + 14, 11); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ink;
    ctx.textAlign = "left";
    ctx.fillText(s.label, lx + 18, 11);
    lx += 18 + ctx.measureText(s.label).width + 14;
  }

  // current-frame marker, and hover crosshair + readout
  const markers = [[Math.min(S.frame, n - 1), false]];
  if (b.hoverFrame != null) markers.push([b.hoverFrame, true]);
  for (const [f, isHover] of markers) {
    const x = Math.round(X(f)) + 0.5;
    ctx.strokeStyle = isHover ? muted : ink;
    ctx.lineWidth = 1;
    ctx.setLineDash(isHover ? [2, 3] : []);
    ctx.beginPath(); ctx.moveTo(x, g.top); ctx.lineTo(x, g.top + ph); ctx.stroke();
    ctx.setLineDash([]);
    for (const s of CURVE_SERIES) {
      const v = st[s.key][f];
      if (v == null) continue;
      ctx.beginPath(); ctx.arc(X(f), Math.max(g.top, Y(v)), 4, 0, 2 * Math.PI);  // pinned at top if above y max
      ctx.fillStyle = s[mode]; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke();
    }
  }
  const fmt = (v) => (v == null ? "–" : `${v.toFixed(2)} px`);
  if (b.hoverFrame == null) {
    // Current frame, one line after the legend.
    const f = Math.min(S.frame, n - 1);
    ctx.fillStyle = muted;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(`frame ${st.frame[f]}: mean ${fmt(st.mean[f])}`, lx + 6, 11);
    return;
  }
  const f = b.hoverFrame;
  const lines = [`frame ${st.frame[f]}  (click to jump)`,
    ...CURVE_SERIES.map((s) => `${s.label}: ${fmt(st[s.key][f])}`), `points: ${st.n_valid[f] ?? 0}`];
  const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 14, th = lines.length * 14 + 8;
  let tx = X(f) + 10;
  if (tx + tw > w - 4) tx = X(f) - 10 - tw;
  const ty = g.top + 2;
  ctx.fillStyle = surface;
  ctx.strokeStyle = grid;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.94;
  ctx.fillRect(tx, ty, tw, th);
  ctx.globalAlpha = 1;
  ctx.strokeRect(tx + 0.5, ty + 0.5, tw - 1, th - 1);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((l, k) => {
    ctx.fillStyle = k === 0 ? muted : ink;
    ctx.fillText(l, tx + 7, ty + 5 + k * 14);
  });
}

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
