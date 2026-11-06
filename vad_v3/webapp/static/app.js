"use strict";
const $ = (id) => document.getElementById(id);
const player = $("player");
const labels = {
    Abuse: "虐待",
    Arrest: "逮捕",
    Arson: "纵火",
    Assault: "袭击",
    Burglary: "入室盗窃",
    Explosion: "爆炸",
    Fighting: "打斗",
    RoadAccidents: "交通事故",
    Robbery: "抢劫",
    Shooting: "枪击",
    Shoplifting: "商店行窃",
    Stealing: "盗窃",
    Vandalism: "破坏",
    unknown: "未分类"
};
const statuses = {uploading: "正在上传", running: "正在检测", completed: "检测完成", failed: "检测失败"};
let file = null, blobURL = null, currentJob = null, pollTimer = null, generation = 0;
let activeRange = null, frameHandle = null, duration = 0, maxUploadMB = 2048, submitting = false;
let configWarning = "";
let historyBusy = false, historyGeneration = 0;
const deletedJobIds = new Set();

function clock(seconds, precise = false) {
    if (!Number.isFinite(seconds)) return "--:--";
    const total = Math.max(0, Math.round(seconds * 1000));
    const s = Math.floor(total / 1000) % 60, m = Math.floor(total / 60000) % 60, h = Math.floor(total / 3600000);
    return (h ? String(h).padStart(2, "0") + ":" : "") + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0") + (precise ? "." + String(total % 1000).padStart(3, "0") : "");
}

function parseTime(text) {
    const value = text.trim();
    if (!/^\d+(?:\.\d+)?$/.test(value) && !/^\d+(?::[0-5]?\d){1,2}(?:\.\d+)?$/.test(value)) return NaN;
    return value.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}

function notify(message, error = false) {
    $("notice").textContent = message;
    $("notice").hidden = !message;
    $("notice").classList.toggle("error", error);
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

async function api(path, options = {}) {
    const response = await fetch(path, {cache: "no-store", ...options});
    const value = await response.json();
    if (!response.ok) throw new Error(value.error || "请求失败，请检查本地服务。");
    return value;
}

function updateDetect() {
    $("detect").disabled = !file || submitting || historyBusy || Boolean(currentJob && ["uploading", "running"].includes(currentJob.status));
}

function clearRange() {
    activeRange = null;
    if (frameHandle !== null) cancelAnimationFrame(frameHandle);
    frameHandle = null;
    document.querySelectorAll(".selected,.timeline-segment.active").forEach(node => node.classList.remove("selected", "active"));
}

function stopAtBoundary() {
    if (!activeRange || player.paused) return;
    if (player.currentTime >= activeRange.end) {
        if ($("loop").checked) {
            player.currentTime = activeRange.start;
            player.play().catch(error => notify(error.message, true));
        } else {
            const end = activeRange.end;
            player.pause();
            player.currentTime = end;
            $("playback-status").textContent = `已播放至区间结束 ${clock(end, true)} · 可再次播放或切换完整视频`;
        }
    }
}

function watchPlayback() {
    frameHandle = null;
    stopAtBoundary();
    if (!player.paused && activeRange) frameHandle = requestAnimationFrame(watchPlayback);
}

function playRange(start, end, index = null) {
    const playableDuration = Number.isFinite(player.duration) ? player.duration : duration;
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= end || end > playableDuration + 0.001) {
        $("playback-status").textContent = `请输入有效区间：0 ≤ 起始 < 结束 ≤ ${clock(playableDuration, true)}`;
        return;
    }
    clearRange();
    activeRange = {start, end: Math.min(end, playableDuration)};
    $("range-start").value = String(start);
    $("range-end").value = String(end);
    if (index !== null) {
        $("result-" + index)?.classList.add("selected");
        $("segment-" + index)?.classList.add("active");
    }
    player.currentTime = start;
    $("playback-status").textContent = `正在播放 ${clock(start, true)} — ${clock(end, true)} · ${$("loop").checked ? "区间循环播放" : "到达末尾自动暂停"}`;
    player.play().then(() => {
        if (frameHandle === null) watchPlayback();
    }).catch(error => {
        $("playback-status").textContent = "无法开始播放：" + error.message;
    });
}

player.addEventListener("play", () => {
    if (activeRange && (player.currentTime < activeRange.start || player.currentTime >= activeRange.end)) player.currentTime = activeRange.start;
    if (frameHandle === null) watchPlayback();
});
player.addEventListener("pause", () => {
    if (frameHandle !== null) cancelAnimationFrame(frameHandle);
    frameHandle = null;
});
player.addEventListener("timeupdate", stopAtBoundary);
player.addEventListener("ended", () => {
    if (activeRange && $("loop").checked) {
        player.currentTime = activeRange.start;
        player.play().catch(() => {
        });
    }
});
player.addEventListener("loadedmetadata", () => {
    duration = player.duration;
    $("duration-label").textContent = clock(duration);
    $("timeline-end").textContent = clock(duration);
    $("range-end").value = String(Math.floor(duration * 1000) / 1000);
    $("play-range").disabled = false;
    $("play-whole").disabled = false;
});
player.addEventListener("error", () => {
    if (!player.getAttribute("src")) return;
    $("video-error").hidden = false;
    $("video-error").textContent = "浏览器无法播放此视频编码。可继续检测；如需区间回看，请转换为 H.264 MP4 后上传。";
    $("play-range").disabled = true;
    $("play-whole").disabled = true;
});

function setVideo(url, name) {
    clearRange();
    player.pause();
    $("video-error").hidden = true;
    $("video-placeholder").hidden = true;
    player.hidden = false;
    $("play-range").disabled = true;
    $("play-whole").disabled = true;
    player.src = url;
    $("range-start").value = "0";
    $("range-end").value = "";
    $("timeline-end").textContent = "--:--";
    $("video-label").textContent = name;
    $("duration-label").textContent = "--:--";
    $("playback-status").textContent = "支持输入秒数或 时:分:秒，例如 12.5 或 01:20";
}

function emptyResults(title, message) {
    const empty = element("div", "empty-results");
    empty.append(element("span", "empty-mark", "⌕"), element("h3", "", title), element("p", "", message));
    $("results").replaceChildren(empty);
    $("timeline").replaceChildren();
    $("result-count").textContent = "0";
    $("download").hidden = true;
}

function selectFile(selected) {
    if (!selected || submitting || historyBusy) return;
    if (!/\.(mp4|webm|mov|m4v|avi|mkv)$/i.test(selected.name)) {
        notify("请选择支持的视频格式。", true);
        return;
    }
    if (selected.size > maxUploadMB * 1024 * 1024) {
        notify(`文件过大，上传上限为 ${maxUploadMB} MB。`, true);
        return;
    }
    clearTimeout(pollTimer);
    generation++;
    currentJob = null;
    file = selected;
    if (blobURL) URL.revokeObjectURL(blobURL);
    blobURL = URL.createObjectURL(file);
    duration = 0;
    setVideo(blobURL, file.name);
    $("file-info").hidden = false;
    $("file-name").textContent = file.name;
    $("file-size").textContent = (file.size / 1024 / 1024).toFixed(1) + " MB";
    $("status-text").textContent = "已选择视频，等待检测";
    $("elapsed").textContent = "";
    $("job-status").className = "job-status";
    $("log-panel").hidden = true;
    $("timeline-hint").textContent = "检测结果将在这里标出";
    emptyResults("视频已就绪", "开始检测后，可查看异常区间及对应的原始描述。");
    notify(configWarning);
    updateDetect();
}

$("video-file").addEventListener("change", event => selectFile(event.target.files[0]));
$("dropzone").addEventListener("keydown", event => {
    if (["Enter", " "].includes(event.key)) {
        event.preventDefault();
        $("video-file").click();
    }
});
for (const type of ["dragenter", "dragover"]) $("dropzone").addEventListener(type, event => {
    event.preventDefault();
    $("dropzone").classList.add("dragover");
});
for (const type of ["dragleave", "drop"]) $("dropzone").addEventListener(type, event => {
    event.preventDefault();
    $("dropzone").classList.remove("dragover");
});
$("dropzone").addEventListener("drop", event => selectFile(event.dataTransfer.files[0]));
$("range-form").addEventListener("submit", event => {
    event.preventDefault();
    playRange(parseTime($("range-start").value), parseTime($("range-end").value));
});
$("play-whole").addEventListener("click", () => {
    clearRange();
    $("playback-status").textContent = "完整视频模式 · 可自由拖动进度条";
});

function renderResults(job) {
    const result = job.result;
    const anomalies = [...result.anomalies].sort((a, b) => a.t_start - b.t_start);
    $("results").replaceChildren();
    $("timeline").replaceChildren();
    $("result-count").textContent = String(anomalies.length);
    $("timeline-hint").textContent = anomalies.length ? "点击色块跳转播放" : "本次检测未发现异常区间";
    if (!anomalies.length) emptyResults("本次检测未发现异常", "仍可输入起止时间回看任意片段。");
    anomalies.forEach((anomaly, index) => {
        const card = element("article", "result-card");
        card.id = "result-" + index;
        const top = element("div", "result-top");
        const category = labels[anomaly.anomaly_type] || anomaly.anomaly_type;
        const play = element("button", "secondary", "▶ 播放片段");
        play.addEventListener("click", () => playRange(anomaly.t_start, anomaly.t_end, index));
        top.append(element("span", "result-number", String(index + 1).padStart(2, "0")), element("span", "category", category + " / " + anomaly.anomaly_type), element("span", "result-times", clock(anomaly.t_start, true) + " → " + clock(anomaly.t_end, true)), element("span", "confidence", "置信度 " + (anomaly.confidence * 100).toFixed(1) + "%"), play);
        card.append(top, element("p", "description-label", "原始 VLM DESCRIPTION"), element("p", "description", anomaly.description || "该结果未记录原始描述；需要使用更新后的 v3 重新检测。"));
        const sources = anomaly.original_descriptions || [];
        if (sources.length) {
            const details = element("details", "source-details");
            details.append(element("summary", "", `${sources.length} 条初次扫描描述 · 查看原始区间`));
            sources.forEach(source => {
                const row = element("div", "source-description");
                row.append(element("span", "", clock(source.t_start, true) + " — " + clock(source.t_end, true)), document.createTextNode(source.description || "未提供描述"));
                details.append(row);
            });
            card.append(details);
        }
        $("results").append(card);
        const marker = element("button", "timeline-segment");
        marker.id = "segment-" + index;
        const total = Number(result.duration);
        marker.style.left = Math.max(0, Math.min(100, anomaly.t_start / total * 100)) + "%";
        marker.style.width = Math.max(0, (Math.min(total, anomaly.t_end) - Math.max(0, anomaly.t_start)) / total * 100) + "%";
        marker.title = `${category} ${clock(anomaly.t_start, true)} — ${clock(anomaly.t_end, true)}`;
        marker.setAttribute("aria-label", marker.title);
        marker.addEventListener("click", () => {
            playRange(anomaly.t_start, anomaly.t_end, index);
            card.scrollIntoView({block: "nearest", behavior: "smooth"});
        });
        $("timeline").append(marker);
    });
    $("download").hidden = false;
    $("download").href = `/api/jobs/${job.id}/result`;
}

async function refreshHistory() {
    const token = ++historyGeneration;
    const {jobs} = await api("/api/jobs");
    if (token !== historyGeneration) return;
    $("history").replaceChildren();
    if (!jobs.length) {
        $("history").append(element("p", "empty-history", "还没有检测记录"));
        return;
    }
    jobs.slice(0, 8).forEach(job => {
        const row = element("div", "history-row");
        const button = element("button", "history-item");
        const detail = element("small", "");
        detail.append(element("span", "", statuses[job.status] || job.status), element("span", "", new Date(job.created_at * 1000).toLocaleString("zh-CN", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit"
        })));
        button.append(element("strong", "", job.filename), detail);
        button.disabled = submitting || historyBusy;
        button.addEventListener("click", () => openJob(job.id).catch(error => notify(error.message, true)));
        const remove = element("button", "history-delete", "删除");
        remove.setAttribute("aria-label", `删除任务 ${job.filename}`);
        remove.disabled = submitting || historyBusy || ["uploading", "running"].includes(job.status);
        remove.title = ["uploading", "running"].includes(job.status) ? "检测完成后可以删除" : "移入回收目录，可撤销";
        remove.addEventListener("click", () => deleteJob(job));
        row.append(button, remove);
        $("history").append(row);
    });
}

function setHistoryBusy(busy) {
    historyBusy = busy;
    if (busy) $("history").querySelectorAll("button").forEach(button => { button.disabled = true; });
    $("refresh-history").disabled = busy;
    updateDetect();
}

function releaseVideo() {
    clearRange();
    player.pause();
    player.removeAttribute("src");
    player.load();
}

function clearDeletedJob() {
    clearTimeout(pollTimer);
    generation++;
    releaseVideo();
    currentJob = null;
    file = null;
    duration = 0;
    if (blobURL) { URL.revokeObjectURL(blobURL); blobURL = null; }
    $("video-file").value = "";
    $("file-info").hidden = true;
    player.hidden = true;
    $("video-placeholder").hidden = false;
    $("video-error").hidden = true;
    $("video-label").textContent = "等待选择视频";
    $("duration-label").textContent = "--:--";
    $("timeline-end").textContent = "--:--";
    $("timeline-hint").textContent = "检测结果将在这里标出";
    $("range-start").value = "0";
    $("range-end").value = "";
    $("play-range").disabled = true;
    $("play-whole").disabled = true;
    $("playback-status").textContent = "请选择视频，或从最近任务打开其他结果";
    $("job-status").className = "job-status";
    $("status-text").textContent = "等待检测";
    $("elapsed").textContent = "";
    $("log-panel").hidden = true;
    $("log").textContent = "";
    emptyResults("任务已删除", "可以撤销删除，或选择其他视频开始检测。");
    updateDetect();
}

async function deleteJob(job) {
    if (submitting || historyBusy) return;
    setHistoryBusy(true);
    const selected = currentJob?.id === job.id;
    if (selected) {
        clearTimeout(pollTimer);
        generation++;
        releaseVideo(); // Release Windows file handles before moving the task.
    }
    try {
        await api(`/api/jobs/${job.id}`, {method: "DELETE"});
        deletedJobIds.add(job.id);
        if (selected) clearDeletedJob();
        if (localStorage.getItem("vad-v3-last-job") === job.id) localStorage.removeItem("vad-v3-last-job");
        notify(`已删除「${job.filename}」，文件已移入本地回收目录。`);
        const undo = element("button", "undo-delete", "撤销");
        undo.addEventListener("click", async () => {
            if (submitting || historyBusy) return;
            undo.disabled = true;
            setHistoryBusy(true);
            try {
                await api(`/api/jobs/${job.id}/restore`, {method: "POST"});
                deletedJobIds.delete(job.id);
                if (!currentJob && !file) emptyResults("任务已恢复", "从最近任务中重新打开该任务。");
                notify(`已恢复「${job.filename}」，可从最近任务重新打开。`);
            } catch (error) {
                notify(error.message, true);
            } finally {
                setHistoryBusy(false);
                await refreshHistory().catch(error => notify(error.message, true));
            }
        });
        $("notice").append(undo);
    } catch (error) {
        if (selected) setVideo(`/api/jobs/${job.id}/video`, job.filename);
        notify(error.message, true);
    } finally {
        setHistoryBusy(false);
        await refreshHistory().catch(error => notify(error.message, true));
    }
}

function displayJob(job) {
    currentJob = job;
    $("status-text").textContent = job.status === "failed" ? job.error : statuses[job.status];
    $("job-status").className = "job-status " + job.status;
    const elapsed = job.processing_time ?? ((job.finished_at || Date.now() / 1000) - (job.started_at || job.created_at));
    $("elapsed").textContent = elapsed >= 0 ? "耗时 " + (elapsed < 1 ? elapsed.toFixed(1) : Math.round(elapsed)) + " s" : "";
    $("log-panel").hidden = !job.log;
    $("log").textContent = job.log || "";
    if (job.status === "completed") renderResults(job);
    if (job.status === "failed") emptyResults("检测未完成", "查看下方日志，处理错误后重新选择视频并提交。");
    updateDetect();
}

async function poll(jobId, token) {
    try {
        const job = await api(`/api/jobs/${jobId}`);
        if (token !== generation) return;
        displayJob(job);
        if (["running", "uploading"].includes(job.status)) pollTimer = setTimeout(() => poll(jobId, token), 2000);
        else refreshHistory().catch(() => {
        });
    } catch (error) {
        if (token !== generation) return;
        notify("无法获取任务状态：" + error.message + " 正在重试。", true);
        pollTimer = setTimeout(() => poll(jobId, token), 4000);
    }
}

async function openJob(id) {
    if (submitting || historyBusy) return;
    clearTimeout(pollTimer);
    const token = ++generation;
    const job = await api(`/api/jobs/${id}`);
    if (token !== generation || historyBusy || deletedJobIds.has(id)) return;
    file = null;
    $("video-file").value = "";
    $("file-info").hidden = true;
    if (blobURL) {
        URL.revokeObjectURL(blobURL);
        blobURL = null;
    }
    duration = job.duration || 0;
    setVideo(`/api/jobs/${id}/video`, job.filename);
    emptyResults("正在读取检测任务", "任务执行期间可以预览视频或手动播放区间。");
    displayJob(job);
    localStorage.setItem("vad-v3-last-job", id);
    if (["running", "uploading"].includes(job.status)) pollTimer = setTimeout(() => poll(id, token), 1500);
}

$("detect").addEventListener("click", () => {
    if (!file || submitting || historyBusy) return;
    submitting = true;
    updateDetect();
    notify("");
    $("upload-progress").hidden = false;
    const form = new FormData();
    form.append("video", file);
    form.append("device", $("device").value);
    form.append("mcts_mode", $("mcts-mode").value);
    form.append("context", $("context").value);
    form.append("full_classification", String($("full-classification").checked));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", event => {
        const percent = event.lengthComputable ? event.loaded / event.total * 100 : 0;
        $("upload-progress").querySelector("progress").value = percent;
        $("upload-progress").querySelector("small").textContent = percent >= 100 ? "上传完成，正在校验视频…" : `正在上传 ${Math.round(percent)}%`;
    });
    xhr.addEventListener("loadend", () => {
        submitting = false;
        $("upload-progress").hidden = true;
        updateDetect();
        if (xhr.status !== 202) {
            notify(xhr.response?.error || "提交失败，请确认服务正在运行。", true);
            return;
        }
        const job = xhr.response;
        clearTimeout(pollTimer);
        const token = ++generation;
        currentJob = job;
        localStorage.setItem("vad-v3-last-job", job.id);
        emptyResults("正在分析视频", "模型加载和检测可能需要几分钟。可以先预览视频，或展开日志查看进度。");
        displayJob(job);
        poll(job.id, token);
        refreshHistory().catch(() => {
        });
    });
    xhr.send(form);
});
$("refresh-history").addEventListener("click", () => refreshHistory().catch(error => notify(error.message, true)));
(async () => {
    try {
        const config = await api("/api/config");
        maxUploadMB = config.max_upload_mb;
        $("upload-limit").textContent = `最大 ${maxUploadMB >= 1024 ? (maxUploadMB / 1024).toFixed(0) + " GB" : maxUploadMB + " MB"} · 文件保存在本机`;
    if (!config.api_key_configured) configWarning = "尚未配置 DASHSCOPE_API_KEY：可以预览视频和播放指定区间；启用检测请在 webapp/.env 中配置密钥后重启服务。";
        notify(configWarning);
        await refreshHistory();
        const last = localStorage.getItem("vad-v3-last-job");
        if (last) await openJob(last).catch(() => localStorage.removeItem("vad-v3-last-job"));
    } catch (error) {
        notify("无法连接本地服务：" + error.message, true);
    }
})();
