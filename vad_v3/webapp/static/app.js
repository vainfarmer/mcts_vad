"use strict";
const $=id=>document.getElementById(id);
const labels={Abuse:"虐待",Arrest:"逮捕",Arson:"纵火",Assault:"袭击",Burglary:"入室盗窃",Explosion:"爆炸",Fighting:"打斗",RoadAccidents:"交通事故",Robbery:"抢劫",Shooting:"枪击",Shoplifting:"商店行窃",Stealing:"盗窃",Vandalism:"破坏",Normal:"正常",unknown:"未分类"};
const statuses={queued:"排队中",running:"检测中",cancelling:"正在取消",completed:"已完成",failed:"失败",interrupted:"已中断",cancelled:"已取消"};
const activeStates=["queued","running","cancelling"], player=$("player");
let selectedVideo=null,selectedFile=null,objectURL=null,currentJob=null,analysisData=null,currentPage="workbench",generation=0;
let activeRange=null,frameHandle=null,taskPage=1,videoPage=1,busy=false,polling=false,reviewSubject=null,reviewVersion=0,reviewScore=1,configInfo={};
let curveChart=null,metricChart=null,confusionChart=null,stageChart=null,treeChart=null;
let experimentSignature="";
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmtTime=v=>Number.isFinite(v)?Math.floor(v/60).toString().padStart(2,"0")+":"+(v%60).toFixed(2).padStart(5,"0"):"--:--";
const date=v=>v?new Date(v*1000).toLocaleString("zh-CN",{hour12:false}):"—";
const pct=v=>typeof v==="number"?(v*100).toFixed(2)+"%":"未计算";
const status=v=>'<span class="status '+esc(v)+'">'+esc(statuses[v]||v)+'</span>';
function button(text,op,id,cls="text-button"){return '<button class="'+cls+'" data-op="'+op+'" data-id="'+esc(id)+'">'+text+'</button>';}
function notify(message,error=false){$("notice").hidden=!message;$("notice").textContent=message;$("notice").classList.toggle("error",error);}
async function api(path,options={}){
    const res=await fetch(path,{cache:"no-store",...options});const data=await res.json();
    if(!res.ok)throw new Error(data.error||"请求失败 "+res.status);return data;
}
const post=(url,data={})=>api(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
function action(fn){return async(...args)=>{try{await fn(...args);}catch(e){notify(e.message,true);}};}
function bind(container,fn){$(container).querySelectorAll("[data-op]").forEach(b=>b.onclick=action(()=>fn(b.dataset.op,b.dataset.id)));}
function switchPage(name){
    currentPage=name;document.querySelectorAll(".page").forEach(p=>p.classList.toggle("active",p.id==="page-"+name));
    document.querySelectorAll(".nav").forEach(p=>p.classList.toggle("active",p.dataset.page===name));
    const loaders={videos:loadVideos,tasks:loadTasks,research:loadResearch,experiments:loadExperiments,settings:loadHealth};
    if(loaders[name])action(loaders[name])();
    setTimeout(()=>{curveChart?.resize();metricChart?.resize();confusionChart?.resize();stageChart?.resize();treeChart?.resize();},0);
}
document.querySelectorAll(".nav").forEach(b=>b.onclick=()=>switchPage(b.dataset.page));
function clearRange(){activeRange=null;if(frameHandle!==null)cancelAnimationFrame(frameHandle);frameHandle=null;}
function parseTime(value){if(!/^\d+(?::[0-5]?\d){0,2}(?:\.\d+)?$/.test(value.trim()))return NaN;return value.trim().split(":").reduce((t,x)=>t*60+Number(x),0);}
function stopAtBoundary(){
    if(!activeRange||player.paused)return;
    if(player.currentTime>=activeRange.end){
        if($("loop").checked){player.currentTime=activeRange.start;player.play().catch(e=>notify(e.message,true));}
        else{player.pause();player.currentTime=activeRange.end;$("playback-status").textContent="已播放至 "+fmtTime(activeRange.end);}
    }
}
function watchPlayback(){frameHandle=null;stopAtBoundary();if(!player.paused&&activeRange)frameHandle=requestAnimationFrame(watchPlayback);}
function playRange(start,end){
    const duration=Number.isFinite(player.duration)?player.duration:currentJob?.duration;
    if(!Number.isFinite(start)||!Number.isFinite(end)||!(0<=start&&start<end&&end<=duration+.05))throw new Error("请输入有效区间：0 ≤ 开始 < 结束 ≤ 视频时长。");
    clearRange();activeRange={start,end:Math.min(end,duration)};
    $("range-start").value=start;$("range-end").value=end;player.currentTime=start;
    $("playback-status").textContent="播放 "+fmtTime(start)+" — "+fmtTime(end);
    player.play().then(watchPlayback).catch(e=>notify(e.message,true));
}
player.addEventListener("play",()=>{if(activeRange&&(player.currentTime<activeRange.start||player.currentTime>=activeRange.end))player.currentTime=activeRange.start;if(frameHandle===null)watchPlayback();});
player.addEventListener("pause",()=>{if(frameHandle!==null)cancelAnimationFrame(frameHandle);frameHandle=null;});
player.addEventListener("timeupdate",stopAtBoundary);
player.addEventListener("loadedmetadata",()=>{$("duration-label").textContent=fmtTime(player.duration);$("range-end").value=player.duration.toFixed(3);$("play-range").disabled=false;});
player.addEventListener("error",()=>{$("video-error").hidden=false;$("video-error").textContent="浏览器无法解码此视频，建议使用 H.264 MP4 预览。检测支持不等于浏览器支持。";});
$("range-form").onsubmit=action(async e=>{e.preventDefault();playRange(parseTime($("range-start").value),parseTime($("range-end").value));});
$("play-whole").onclick=()=>{clearRange();$("playback-status").textContent="已切换完整视频播放。";};
function setVideo(url,name){clearRange();player.pause();player.src=url;player.hidden=false;$("video-placeholder").hidden=true;$("video-error").hidden=true;$("player-name").textContent=name;$("play-range").disabled=true;}
function resetResults(){ $("results").className="empty";$("results").textContent="视频已就绪，可提交新的检测任务。";$("result-count").textContent="0";$("timeline").replaceChildren();$("export-links").hidden=true;$("job-actions").replaceChildren();$("job-status").textContent="等待提交";$("log").textContent="尚无日志";$("job-stage").textContent="";$("elapsed").textContent="";}
// 从历史任务切换到新视频时恢复新任务默认策略，避免把历史 strict 无意带入新任务。
function resetMctsDefaults(){
    $("mcts-mode").value=configInfo.defaults?.mcts_mode||"refine";
    $("acceptance-policy").value=configInfo.defaults?.mcts_acceptance_policy||"force_best";
}
function chooseFile(file){
    if(!file)return;if(file.size>(configInfo.max_upload_mb||2048)*1024*1024)throw new Error("视频超过上传大小限制。");
    if(currentJob)resetMctsDefaults();
    selectedFile=file;selectedVideo=null;currentJob=null;analysisData=null;generation++;
    if(objectURL)URL.revokeObjectURL(objectURL);objectURL=URL.createObjectURL(file);setVideo(objectURL,file.name);
    $("selected-video").textContent=file.name;$("video-meta").textContent=(file.size/1024/1024).toFixed(1)+" MB · 待上传";
    $("detect").disabled=false;resetResults();switchPage("workbench");notify("");
}
$("choose-video").onclick=$("library-upload").onclick=$("dropzone").onclick=()=>$("file-input").click();
$("dropzone").onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();$("file-input").click();}};
$("file-input").onchange=action(async e=>{chooseFile(e.target.files[0]);e.target.value="";});
$("dropzone").ondragover=e=>{e.preventDefault();$("dropzone").classList.add("drag");};
$("dropzone").ondragleave=()=>$("dropzone").classList.remove("drag");
$("dropzone").ondrop=action(async e=>{e.preventDefault();$("dropzone").classList.remove("drag");chooseFile(e.dataTransfer.files[0]);});
function taskConfig(){return {device:$("device").value,mcts_mode:$("mcts-mode").value,context:$("context").value,full_classification:$("full-classification").checked,mcts:{acceptance_policy:$("acceptance-policy").value,search_strategy:$("strategy").value,max_vlm_evaluations_per_video:Number($("budget").value)},postprocess:{qscore_enabled:$("qscore").checked,bue_enabled:$("bue").checked,cbt_enabled:$("cbt").checked,final_nms_enabled:$("final-nms").checked}};}
$("detect").onclick=action(async()=>{
    if(busy)return;busy=true;$("detect").disabled=true;
    try{
        if(!selectedVideo){if(!selectedFile)throw new Error("请先选择视频。");notify("正在上传并验证视频…");const fd=new FormData();fd.append("video",selectedFile);selectedVideo=await api("/api/videos",{method:"POST",body:fd});}
        const job=await api("/api/jobs",{method:"POST",headers:{"Content-Type":"application/json","Idempotency-Key":crypto.randomUUID()},body:JSON.stringify({video_id:selectedVideo.id,config:taskConfig()})});
        notify("任务已进入队列，可以继续导入其他视频。");await openJob(job.id);await loadRecent();
    }finally{busy=false;$("detect").disabled=!selectedVideo&&!selectedFile;}
});
async function loadRecent(){
    const data=await api("/api/jobs");
    $("recent-jobs").innerHTML=data.jobs.slice(0,5).map(j=>'<button class="recent-item" data-op="open" data-id="'+j.id+'"><span>'+esc(j.filename)+'</span><small>'+esc(statuses[j.status])+' · '+esc(j.mcts_mode)+'</small></button>').join("")||'<p class="hint">暂无检测任务</p>';
    bind("recent-jobs",(_,id)=>openJob(id));
}
async function openJob(id){
    const token=++generation;const job=await api("/api/jobs/"+id);if(token!==generation)return;
    currentJob=job;analysisData=null;selectedFile=null;selectedVideo={id:job.video_id,filename:job.filename};localStorage.setItem("vad-current-job",id);
    $("selected-video").textContent=job.filename;$("video-meta").textContent="历史输入 · "+fmtTime(job.duration);
    $("device").value=job.device;$("mcts-mode").value=job.mcts_mode;$("detect").disabled=false;
    $("acceptance-policy").value=job.config?.detector_config?.mcts?.acceptance_policy||"strict";
    setVideo("/api/jobs/"+id+"/video",job.filename);switchPage("workbench");renderJob(job);
}
function renderJob(job){
    $("job-status").className="status "+job.status;$("job-status").textContent=statuses[job.status]||job.status;
    const elapsed=job.attempts?.at(-1)?.metrics?.elapsed_seconds;$("elapsed").textContent=elapsed?elapsed.toFixed(1)+" s":"";
    $("log").textContent=job.log||"尚无执行日志";$("job-stage").textContent=job.status==="queued"?"等待 Worker 领取":job.error?"请查看执行日志":"";
    $("job-actions").innerHTML=activeStates.includes(job.status)?button("取消任务","cancel",job.id,"secondary"):["failed","interrupted","cancelled"].includes(job.status)?button("重新运行","retry",job.id,"secondary"):"";
    bind("job-actions",async(op,id)=>{await post("/api/jobs/"+id+"/"+op);await pollJob();});
    if(job.result){renderResults(job.result);$("export-links").hidden=false;$("export-links").innerHTML='<a href="/api/jobs/'+job.id+'/result">JSON</a><a href="/api/jobs/'+job.id+'/export?format=csv">CSV</a><a href="/api/jobs/'+job.id+'/export">分析报告 ↗</a>';}
    else{$("results").className="empty";$("results").textContent=job.error?"任务未完成，不能据此判断视频正常。":job.status==="queued"?"任务已排队，可离开页面或继续提交。":"正在处理，请等待检测结果。";$("timeline").replaceChildren();$("result-count").textContent="0";$("export-links").hidden=true;}
}
function renderResults(result){
    const rows=result.anomalies;$("result-count").textContent=rows.length;$("results").className="";
    $("results").innerHTML=rows.map((d,i)=>'<article class="result-card"><div class="result-top"><span class="mono">'+String(i+1).padStart(2,"0")+'</span><span class="category">'+esc(labels[d.anomaly_type]||d.anomaly_type)+'</span><span class="result-times">'+fmtTime(d.t_start)+" — "+fmtTime(d.t_end)+'</span><span class="hint">评分 '+d.confidence.toFixed(3)+'</span>'+button("▶ 回看","play",i)+button("复核","review",i)+'</div><p class="description-label">发现阶段原始描述 / ORIGINAL DESCRIPTION</p><p class="description">'+esc(d.description||"未记录原始描述")+'</p>'+(d.original_descriptions?.length?'<details class="source-details"><summary>查看 '+d.original_descriptions.length+' 条来源区间</summary>'+d.original_descriptions.map(s=>'<div class="source-description"><b>'+fmtTime(s.t_start)+" — "+fmtTime(s.t_end)+'</b>'+esc(s.description)+'</div>').join("")+'</details>':"")+'</article>').join("")||'<p class="empty">本次检测未检出异常。该结论不等同于人工确认视频正常。</p>';
    $("timeline").innerHTML=rows.map((d,i)=>'<button class="timeline-segment" data-op="play" data-id="'+i+'" title="'+esc(labels[d.anomaly_type]||d.anomaly_type)+" "+fmtTime(d.t_start)+'" style="left:'+(100*d.t_start/result.duration)+'%;width:'+(100*(d.t_end-d.t_start)/result.duration)+'%"></button>').join("");
    const handle=(op,id)=>{const d=rows[Number(id)];return op==="play"?playRange(d.t_start,d.t_end):openReview("det-"+id,d);};
    bind("results",handle);bind("timeline",handle);
}
async function pollJob(){
    if(!currentJob||polling)return;polling=true;const id=currentJob.id,token=generation;
    try{const job=await api("/api/jobs/"+id);if(token!==generation)return;currentJob=job;renderJob(job);
        if(activeStates.includes(job.status)){const events=await api("/api/jobs/"+id+"/events");if(token!==generation)return;const latest=events.events.at(-1);if(latest)$("job-stage").textContent=latest.stage+" · "+latest.type;}
    }finally{polling=false;}
}
async function loadVideos(){
    const {videos}=await api("/api/videos?q="+encodeURIComponent($("video-search").value)+"&page="+videoPage+"&deleted="+($("video-trash").checked?1:0));$("video-page").textContent=videoPage;
    $("video-list").innerHTML=videos.map(v=>'<article class="video-card"><div class="video-icon">▷</div><h3>'+esc(v.filename)+'</h3><p>'+fmtTime(v.metadata.duration)+" · "+(v.metadata.width||"—")+" × "+(v.metadata.height||"—")+" · "+Number(v.metadata.fps||0).toFixed(2)+' FPS</p><p>'+esc(date(v.created_at))+'</p><div class="buttons">'+(v.deleted_at?button("恢复","restore",v.id):button("打开 / 检测","select",v.id,"secondary")+button("删除","delete",v.id))+'</div></article>').join("")||'<div class="panel empty">暂无视频</div>';
    bind("video-list",async(op,id)=>{
        if(op==="select"){if(currentJob)resetMctsDefaults();const v=videos.find(x=>x.id===id);selectedVideo=v;selectedFile=null;currentJob=null;analysisData=null;generation++;setVideo("/api/videos/"+v.id+"/stream",v.filename);$("selected-video").textContent=v.filename;$("video-meta").textContent="已保存 · "+fmtTime(v.metadata.duration);$("detect").disabled=false;resetResults();switchPage("workbench");}
        else{if(op==="delete")await api("/api/videos/"+id,{method:"DELETE"});else await post("/api/videos/"+id+"/restore");await loadVideos();notify(op==="delete"?"视频已移入回收站，历史任务仍可回看。":"视频已恢复。");}
    });
}
async function loadTasks(){
    const {jobs}=await api("/api/jobs?status="+$("task-filter").value+"&q="+encodeURIComponent($("task-search").value)+"&page="+taskPage+"&deleted="+($("task-trash").checked?1:0));$("task-page").textContent=taskPage;
    $("task-list").innerHTML=jobs.map(j=>'<tr><td>'+esc(j.filename)+'<small>'+j.id.slice(0,12)+'</small></td><td>'+status(j.status)+'</td><td>'+esc(j.mcts_mode)+" · "+esc(j.device)+'</td><td>'+esc(date(j.created_at))+'</td><td>'+(j.deleted_at?button("恢复","restore",j.id):button("打开","open",j.id)+(activeStates.includes(j.status)?button("取消","cancel",j.id):(["failed","interrupted","cancelled"].includes(j.status)?button("重试","retry",j.id):"")+button("删除","delete",j.id)))+'</td></tr>').join("")||'<tr><td colspan="5" class="empty">暂无任务</td></tr>';
    bind("task-list",async(op,id)=>{
        if(op==="open")return openJob(id);
        if(op==="delete"){await api("/api/jobs/"+id,{method:"DELETE"});if(currentJob?.id===id){currentJob=null;analysisData=null;generation++;clearRange();player.pause();player.removeAttribute("src");player.load();localStorage.removeItem("vad-current-job");resetResults();}notify("任务已移入回收站，可恢复。");}
        else await post("/api/jobs/"+id+"/"+op);await loadTasks();await loadRecent();
    });
}
$("refresh-tasks").onclick=action(loadTasks);$("refresh-videos").onclick=action(loadVideos);
for(const id of ["task-filter","task-trash","task-search"])$(id).onchange=action(async()=>{taskPage=1;await loadTasks();});
for(const id of ["video-search","video-trash"])$(id).onchange=action(async()=>{videoPage=1;await loadVideos();});
$("task-prev").onclick=action(async()=>{taskPage=Math.max(1,taskPage-1);await loadTasks();});$("task-next").onclick=action(async()=>{taskPage++;await loadTasks();});
$("video-prev").onclick=action(async()=>{videoPage=Math.max(1,videoPage-1);await loadVideos();});$("video-next").onclick=action(async()=>{videoPage++;await loadVideos();});
async function loadResearch(){
    const ready=!!currentJob?.result;$("research-empty").hidden=ready;$("research-content").hidden=!ready;$("manual-event").disabled=!ready;if(!ready)return;
    const id=currentJob.id,token=generation;const [data,chartData]=await Promise.all([api("/api/jobs/"+id+"/analysis"),api("/api/jobs/"+id+"/curves")]);if(token!==generation)return;analysisData=data;
    $("research-label").textContent=currentJob.filename+" · "+id.slice(0,12)+" · "+(data.recorded?"已记录过程证据":"历史任务，部分过程未记录");
    const runtime=data.runtime||{},usage=data.usage||{};
    $("execution-summary").innerHTML='<p>退出原因：'+esc(data.exit_reason)+'</p><p>总耗时 '+(runtime.elapsed_seconds?.toFixed(2)??"未记录")+' 秒 · 峰值内存 '+(runtime.peak_ram_bytes?(runtime.peak_ram_bytes/1048576).toFixed(1)+" MB":"未记录")+' · 峰值显存 '+(runtime.peak_gpu_bytes?(runtime.peak_gpu_bytes/1048576).toFixed(1)+" MB":"未记录")+'</p><p>实际请求 '+esc(usage.requests??"未记录")+' 次 · 总 Token '+esc(usage.total_tokens??"未记录")+'</p><details><summary>查看分阶段耗时与请求用量</summary><pre>'+esc(JSON.stringify({stage_seconds:data.stage_seconds,usage:data.usage},null,2))+'</pre></details>';
    $("classification-detail").textContent=JSON.stringify({classification:data.classification??"此任务未记录",effective_config:data.config},null,2);
    const qscore=data.observations?.qscore;
    $("qscore-detail").innerHTML=qscore?'<table><thead><tr><th>区间（秒）</th><th>边界锐度</th><th>峰值对比</th><th>中心匹配</th><th>归一化质量</th><th>评分前 → 后</th></tr></thead><tbody>'+qscore.components.map(r=>'<tr><td>'+r.t_start.toFixed(3)+" — "+r.t_end.toFixed(3)+'</td><td>'+(r.sharpness?.toFixed(4)??"无采样")+'</td><td>'+(r.contrast?.toFixed(4)??"—")+'</td><td>'+(r.center_fit?.toFixed(4)??"—")+'</td><td>'+r.normalized_quality.toFixed(4)+'</td><td>'+r.before.toFixed(4)+" → "+r.after.toFixed(4)+'</td></tr>').join("")+'</tbody></table>':'<p class="empty">此任务未记录 Q-score 分量。</p>';
    curveChart??=echarts.init($("curve-chart"));curveChart.setOption({color:["#28795d","#bb945b","#668eb0","#b6bc79"],tooltip:{trigger:"axis",renderMode:"richText",valueFormatter:v=>Number(v).toFixed(3)},legend:{type:"scroll",textStyle:{fontSize:10}},grid:{left:45,right:20,top:45,bottom:50},xAxis:{type:"value",name:"秒"},yAxis:{type:"value"},dataZoom:[{type:"inside"},{type:"slider",height:16,bottom:10}],series:chartData.curves.map(c=>({name:c.name,type:"line",showSymbol:false,sampling:"minmax",data:c.timestamps.map((t,i)=>[t,c.values[i]])}))},true);
    curveChart.off("click");curveChart.on("click",e=>{if(Array.isArray(e.value)){switchPage("workbench");player.currentTime=e.value[0];}});
    $("curve-note").textContent=chartData.curves.length?"曲线保留各自时间轴；融合评分不一定与末尾处理后的区间重合。":"此任务未记录可用曲线。";
    const stages=Object.entries(data.stage_trace||{}).sort(([a],[b])=>(data.stage_order||[]).indexOf(a)-(data.stage_order||[]).indexOf(b));
    drawStageChart(stages);
    const tracedCandidates=Object.entries(data.mcts?.candidates||{}).filter(([,c])=>c.nodes?.length);
    $("tree-choice-label").hidden=$("tree-chart").hidden=!tracedCandidates.length;
    $("tree-note").textContent=tracedCandidates.length?"节点来自实际搜索记录；点击节点查看统计与奖励，支持展开、缩放。":"此任务没有可恢复的完整树节点记录。";
    $("tree-play").hidden=true;$("tree-detail").textContent="";
    $("tree-choice").innerHTML=tracedCandidates.map(([key])=>'<option value="'+esc(key)+'">候选 '+esc(key)+'</option>').join("");
    $("tree-choice").onchange=()=>drawSearchTree(data.mcts.candidates[$("tree-choice").value]);
    if(tracedCandidates.length)drawSearchTree(tracedCandidates[0][1]);
    $("stage-table").innerHTML=stages.length?'<table><thead><tr><th>阶段</th><th>来源 / 类别</th><th>开始</th><th>结束</th><th>评分</th></tr></thead><tbody>'+stages.flatMap(([name,rows])=>rows.length?rows.map(d=>'<tr><td>'+esc(name)+'</td><td>'+esc(d.source_candidate_ids?.join(",")??"—")+" / "+esc(labels[d.anomaly_type]||d.anomaly_type||"—")+'</td><td>'+Number(d.t_start).toFixed(3)+'</td><td>'+Number(d.t_end).toFixed(3)+'</td><td>'+Number(d.confidence||0).toFixed(4)+'</td></tr>'):['<tr><td>'+esc(name)+'</td><td colspan="4">无存活候选</td></tr>']).join("")+'</tbody></table>':'<p class="empty">此任务未记录阶段快照。</p>';
    $("evidence-list").innerHTML=data.evidence.map(r=>'<div class="evidence-row"><b>'+esc(r.id)+" · "+esc(r.scope?.stage||"VLM")+'</b><div class="evidence-images">'+r.images.map(i=>'<div class="evidence-image"><img loading="lazy" src="/api/jobs/'+id+'/evidence/'+encodeURIComponent(i.file.split("/").at(-1))+'" alt="实际 VLM 输入图像"><span>'+(i.source_times?.length===1?fmtTime(i.source_times[0]):"时间未唯一记录")+'</span></div>').join("")+'</div><details><summary>查看原始提示与响应</summary><pre>'+esc(r.prompt)+'</pre><pre>'+esc(r.response||r.error||"请求未完成")+'</pre></details></div>').join("")||'<p class="empty">此任务未保存实际输入图像，不生成替代证据。</p>';
    $("mcts-list").innerHTML=Object.entries(data.mcts?.candidates||{}).map(([key,c])=>'<details class="evidence-row"><summary>候选 '+esc(key)+" · "+esc(c.reason||c.skipped_reason||"未记录")+" · "+(c.logical_requests??0)+' 次逻辑请求</summary><p class="hint">原边界 '+esc(c.baseline?.join(" — "))+" / 提案 "+esc(c.proposed?.join(" — ")||"无")+'</p>'+(c.nodes?'<table><thead><tr><th>节点 / 父节点</th><th>区间</th><th>访问次数</th><th>平均价值</th><th>奖励</th></tr></thead><tbody>'+c.nodes.map(n=>'<tr><td>'+n.node_id+" / "+(n.parent_id??(n.depth===0?"根":"父节点未记录"))+'</td><td>'+esc(n.interval.join(" — "))+'</td><td>'+n.visits+'</td><td>'+(n.mean_value?.toFixed(4)??"—")+'</td><td>'+(n.evaluation?.reward?.toFixed(4)??"无有效评价")+'</td></tr>').join("")+'</tbody></table>':'<p class="hint">历史轨迹未记录完整父子关系，仅展示评价序列。</p>')+'<pre>'+esc(JSON.stringify(c.evaluations||[],null,2))+'</pre></details>').join("")||'<p class="empty">'+esc(data.mcts?.skipped_reason||"未运行或未记录 MCTS 搜索。")+'</p>';
    renderReviews(data.reviews);setTimeout(()=>curveChart.resize(),0);
}
$("curve-export").onclick=()=>{if(!curveChart)return;const a=document.createElement("a");a.download="vad-curves.png";a.href=curveChart.getDataURL({pixelRatio:3,backgroundColor:"#fff"});a.click();};
function drawStageChart(stages){
    const rows=stages.flatMap(([name,items])=>items.filter(d=>Number.isFinite(d.t_start)&&Number.isFinite(d.t_end)).map(d=>({stage:name,event:d})));
    $("stage-chart").hidden=!rows.length;if(!rows.length)return;
    $("stage-chart").style.height=Math.min(900,Math.max(280,rows.length*21+70))+"px";
    stageChart??=echarts.init($("stage-chart"));
    stageChart.setOption({tooltip:{renderMode:"richText",formatter:p=>rows[p.dataIndex].stage+"\n"+p.value[1].toFixed(3)+" — "+p.value[2].toFixed(3)+" 秒"},grid:{left:145,right:25,top:15,bottom:40},xAxis:{type:"value",min:0,name:"秒"},yAxis:{type:"category",inverse:true,data:rows.map(r=>r.stage),axisLabel:{fontSize:9}},series:[{type:"custom",encode:{x:[1,2],y:0},data:rows.map((r,i)=>[i,r.event.t_start,r.event.t_end]),renderItem:(params,api)=>{const y=api.value(0),start=api.coord([api.value(1),y]),end=api.coord([api.value(2),y]);return {type:"rect",shape:{x:start[0],y:start[1]-6,width:Math.max(2,end[0]-start[0]),height:12},style:{fill:"#43896d"}};}}]},true);
    stageChart.off("click");stageChart.on("click",p=>{const d=rows[p.dataIndex].event;switchPage("workbench");playRange(d.t_start,d.t_end);});stageChart.resize();
}
function drawSearchTree(candidate){
    if(!candidate?.nodes?.length)return;
    function selectNode(record){$("tree-detail").textContent=JSON.stringify(record,null,2);$("tree-play").hidden=false;$("tree-play").onclick=()=>{switchPage("workbench");playRange(...record.interval);};}
    $("tree-node-choice").innerHTML=candidate.nodes.map(n=>'<option value="'+n.node_id+'">#'+n.node_id+" · "+n.interval.map(t=>t.toFixed(2)).join(" — ")+'</option>').join("");
    $("tree-node-choice").onchange=()=>selectNode(candidate.nodes.find(n=>String(n.node_id)===$("tree-node-choice").value));
    const records=new Map(candidate.nodes.map(n=>[n.node_id,{name:"#"+n.node_id+" · "+n.interval.map(x=>x.toFixed(2)).join("—"),record:n,children:[]}]));
    const roots=[];candidate.nodes.forEach(n=>{const item=records.get(n.node_id),parent=records.get(n.parent_id);if(parent)parent.children.push(item);else roots.push(item);});
    const root=roots.length===1?roots[0]:{name:"已记录节点（部分父节点缺失）",children:roots};
    treeChart??=echarts.init($("tree-chart"));
    treeChart.setOption({tooltip:{renderMode:"richText",formatter:p=>p.data.record?p.name+"\n访问 "+p.data.record.visits+" 次 · 平均价值 "+(p.data.record.mean_value??0).toFixed(4):p.name},series:[{type:"tree",data:[root],top:20,bottom:20,left:100,right:150,initialTreeDepth:3,roam:true,symbolSize:12,label:{fontSize:11,color:"#315748"},lineStyle:{color:"#a4c3ad"},itemStyle:{color:"#28795d"},expandAndCollapse:true}]},true);
    treeChart.off("click");treeChart.on("click",p=>{if(p.data.record){$("tree-node-choice").value=String(p.data.record.node_id);selectNode(p.data.record);}});treeChart.resize();
}
function renderReviews(rows){
    const latest={};rows.forEach(r=>latest[r.subject_id]=r);
    $("review-list").innerHTML=rows.map(r=>'<div class="review-row"><b>'+esc(r.subject_id)+" · v"+r.revision+" · "+esc(r.action)+'</b>'+(latest[r.subject_id]===r&&r.after?button("继续修订","edit",r.subject_id):"")+'<p>'+esc(r.reason||"未填写备注")+" · "+esc(date(r.created_at))+'</p><small>'+esc(r.after?r.after.anomaly_type+" / "+r.after.t_start+" — "+r.after.t_end+" 秒":"已标记误报")+'</small></div>').join("")||'<p class="empty">暂无人工复核记录。</p>';
    bind("review-list",(_,id)=>openReview(id,latest[id].after));
}
async function openReview(subject,row){
    if(!currentJob?.result)throw new Error("请打开已完成的任务。");analysisData=await api("/api/jobs/"+currentJob.id+"/analysis");
    const prior=analysisData.reviews.filter(r=>r.subject_id===subject).at(-1);reviewSubject=subject;reviewVersion=prior?.revision||0;row=prior?.after||row;reviewScore=row?.confidence??1;
    $("review-title").textContent=subject?"复核检测区间":"补充漏检事件";$("review-action").value=subject?"corrected":"added";$("review-category").value=row?.anomaly_type||"Fighting";
    $("review-start").value=row?.t_start??0;$("review-end").value=row?.t_end??Math.min(5,currentJob.duration);$("review-reason").value="";$("review-error").textContent="";$("review-dialog").showModal();
}
$("manual-event").onclick=action(()=>openReview(null,null));$("review-close").onclick=()=>$("review-dialog").close();
$("review-form").onsubmit=async e=>{e.preventDefault();try{
    await post("/api/jobs/"+currentJob.id+"/reviews",{subject_id:reviewSubject,expected_revision:reviewVersion,action:$("review-action").value,reason:$("review-reason").value,after:{t_start:Number($("review-start").value),t_end:Number($("review-end").value),anomaly_type:$("review-category").value,confidence:reviewScore}});
    $("review-dialog").close();notify("复核已保存，模型原始结果保持不变。");if(currentPage==="research")await loadResearch();
}catch(error){$("review-error").textContent=error.message;}};
async function loadExperiments(){
    const data=await api("/api/experiments");
    const signature=JSON.stringify(data);
    if(signature===experimentSignature)return;
    experimentSignature=signature;
    const selectedProtocols={};
    $("experiment-list").querySelectorAll("select").forEach(s=>selectedProtocols[s.id]=s.value);
    $("experiment-list").innerHTML=data.experiments.map(e=>'<article class="experiment-card"><h3>'+esc(e.name)+'</h3><p>'+e.manifest.video_count+" 个视频 · "+esc(e.manifest.groups.join(" / "))+" · "+esc(date(e.created_at))+'</p><div class="toolbar"><select id="protocol-'+e.id+'"><option value="vadclip_native_fps">源视频 FPS · 类别相关 AP</option><option value="report_historical_v1">历史固定 30 FPS 协议</option></select>'+button("运行离线评价","evaluate",e.id,"secondary")+'</div>'+e.evaluations.map(r=>'<div class="evaluation-row">'+status(r.status)+'<small>'+esc(r.protocol.name)+" · "+esc(date(r.created_at))+'</small>'+(r.status==="completed"?button("查看结果 ↗","metrics",r.id):r.error?'<span class="error-text">'+esc(r.error)+'</span>':"")+'</div>').join("")+'</article>').join("")||'<div class="panel empty">暂无实验。导入历史预测，或上传包含 GT 与各组预测的实验包。</div>';
    $("experiment-list").querySelectorAll("select").forEach(s=>{if(selectedProtocols[s.id])s.value=selectedProtocols[s.id];});
    bind("experiment-list",async(op,id)=>{if(op==="metrics")return showMetrics(id);await post("/api/experiments/"+id+"/evaluate",{protocol:$("protocol-"+id).value});await loadExperiments();});
}
$("import-history").onclick=action(async()=>{$("import-history").disabled=true;try{notify("正在核对历史预测与标注…");await post("/api/experiments/import-history");notify("历史实验已导入，可选择协议重新评价。");await loadExperiments();}finally{$("import-history").disabled=false;}});
$("import-experiment").onclick=()=>$("experiment-file").click();
$("experiment-from-jobs").onclick=action(async()=>{
    const {jobs}=await api("/api/jobs?status=completed");
    $("experiment-job-choices").innerHTML=jobs.map(j=>'<label>'+esc(j.filename)+' · '+esc(j.mcts_mode)+' · '+j.id.slice(0,8)+'<select data-experiment-job="'+j.id+'"><option value="">不选择</option><option value="A">配置 A</option><option value="B">配置 B</option></select></label>').join("")||'<p class="empty">没有已完成任务。</p>';
    $("experiment-error").textContent="";$("experiment-dialog").showModal();
});
$("experiment-close").onclick=()=>$("experiment-dialog").close();
$("experiment-form").onsubmit=async e=>{
    e.preventDefault();try{
        const groups={A:[],B:[]};
        document.querySelectorAll("[data-experiment-job]").forEach(s=>{if(s.value)groups[s.value].push(s.dataset.experimentJob);});
        if(!groups.A.length||!groups.B.length)throw new Error("请为两个配置组分别选择任务。");
        const ground_truth=JSON.parse(await $("gt-file").files[0].text());
        await post("/api/experiments/from-jobs",{name:$("experiment-name").value,groups,ground_truth});
        $("experiment-dialog").close();notify("任务实验已创建，选择评价协议后开始计算。");await loadExperiments();
    }catch(error){$("experiment-error").textContent=error.message;}
};
$("experiment-file").onchange=action(async e=>{const file=e.target.files[0];if(!file)return;const fd=new FormData();fd.append("bundle",file);await api("/api/experiments",{method:"POST",body:fd});e.target.value="";notify("实验包已校验并保存。");await loadExperiments();});
async function showMetrics(id){
    const record=await api("/api/evaluations/"+id);if(!record.metrics)throw new Error(record.error||"评价尚未完成");
    const m=record.metrics,groups=Object.entries(m.groups);$("metric-view").hidden=false;
    $("cost-table").innerHTML='<table><thead><tr><th>配置组</th><th>耗时中位数 / P95（秒）</th><th>耗时覆盖</th><th>已知 Token</th><th>用量覆盖</th></tr></thead><tbody>'+groups.map(([name,g])=>'<tr><td>'+esc(name)+'</td><td>'+(g.cost?.median_seconds?.toFixed(2)??"未记录")+" / "+(g.cost?.p95_seconds?.toFixed(2)??"未记录")+'</td><td>'+(g.cost?.timing_coverage??"—")+'</td><td>'+(g.cost?.total_tokens??"未记录")+'</td><td>'+(g.cost?.usage_coverage??"—")+'</td></tr>').join("")+'</tbody></table><p class="hint">配对实验共享上游；总用量不可解释为各分支独立运行成本。未知用量不按零消耗处理。</p>';
    $("metric-protocol").textContent=m.protocol.name+" · 13 类宏平均 · IoU 0.1 / 0.2 / 0.3 / 0.4 / 0.5 · 覆盖 "+m.coverage.completed+"/"+m.coverage.expected;
    $("metric-downloads").innerHTML='<a href="/api/evaluations/'+id+'/download">指标 JSON</a> · <a href="/api/evaluations/'+id+'/download?format=csv">指标 CSV</a> · <a href="/api/evaluations/'+id+'/download?audit=1">匹配审计</a>';
    $("metric-table").innerHTML='<table><thead><tr><th>配置组</th><th>异常视频 AVG</th><th>全视频 AVG</th><th>帧 ROC-AUC</th><th>视频分类准确率</th><th>正常误报率</th></tr></thead><tbody>'+groups.map(([name,g])=>'<tr><td>'+esc(name)+'</td><td>'+pct(g.abnormal.values.AVG)+'</td><td>'+pct(g.all_videos.values.AVG)+'</td><td title="'+esc(g.frame.reason||"")+'">'+pct(g.frame.auc)+'</td><td>'+pct(g.classification.accuracy)+'</td><td>'+pct(g.classification.normal_false_positive_rate)+'</td></tr>').join("")+'</tbody></table>';
    metricChart??=echarts.init($("metric-chart"));metricChart.setOption({color:["#28795d","#c6a36d","#7ba2b6"],tooltip:{trigger:"axis",renderMode:"richText",valueFormatter:v=>Number(v).toFixed(3)},legend:{},grid:{left:50,right:25,bottom:35},xAxis:{type:"category",data:m.protocol.thresholds.map(x=>"IoU "+x)},yAxis:{type:"value",name:"mAP (%)"},series:groups.map(([name,g])=>({name,type:"bar",data:m.protocol.thresholds.map(t=>g.abnormal.values["mAP@"+t.toFixed(1)]*100)}))},true);
    $("class-metrics").innerHTML='<table><thead><tr><th>类别</th>'+groups.map(([n])=>'<th>'+esc(n)+' · AP@0.5</th>').join("")+'</tr></thead><tbody>'+configInfo.categories.map(c=>'<tr><td>'+esc(labels[c])+'</td>'+groups.map(([,g])=>'<td>'+pct(g.abnormal.per_class[c]["AP@0.5"])+'</td>').join("")+'</tr>').join("")+'</tbody></table>';
    $("video-errors").innerHTML=groups.map(([name,g])=>'<details><summary>'+esc(name)+' · 视频分类错误案例</summary><table><thead><tr><th>视频</th><th>真实类别</th><th>预测类别</th><th>区间数</th><th>帧 AUC</th></tr></thead><tbody>'+Object.entries(g.per_video||{}).filter(([,v])=>!v.correct).map(([vid,v])=>'<tr><td>'+esc(vid)+'</td><td>'+esc(labels[v.actual]||v.actual)+'</td><td>'+esc(labels[v.predicted]||v.predicted)+'</td><td>'+v.detection_count+'</td><td>'+pct(v.auc)+'</td></tr>').join("")+'</tbody></table></details>').join("");
    const [name,g]=groups[0],cls=g.classification,matrix=cls.confusion_matrix.flatMap((row,i)=>row.map((v,j)=>[j,i,v]));
    confusionChart??=echarts.init($("confusion-chart"));confusionChart.setOption({title:{text:name+" · 视频分类混淆矩阵",textStyle:{fontSize:12,color:"#55725d"}},tooltip:{renderMode:"richText"},grid:{left:95,right:70,top:40,bottom:80},xAxis:{type:"category",data:cls.labels.map(c=>labels[c]||c),axisLabel:{rotate:40,fontSize:9},name:"预测"},yAxis:{type:"category",data:cls.labels.map(c=>labels[c]||c),axisLabel:{fontSize:9},name:"真实"},visualMap:{min:0,max:Math.max(1,...matrix.map(x=>x[2])),orient:"vertical",right:0,top:55,inRange:{color:["#f4f7ec","#28795d"]}},series:[{type:"heatmap",data:matrix,label:{show:true,fontSize:8}}]},true);
    $("metric-view").scrollIntoView({behavior:"smooth",block:"start"});metricChart.resize();confusionChart.resize();
}
async function loadHealth(){
    const health=await api("/api/health");$("service-status").textContent=health.worker_online?"● Web 与 Worker 在线":"○ Web 在线 · Worker 未就绪";$("service-status").classList.toggle("online",health.worker_online);
    $("health-details").innerHTML="<p>Web：在线 · Worker："+(health.worker_online?"在线":"离线 / 尚未启动")+"</p><p>VLM："+esc(configInfo.model||"—")+" · 密钥："+(configInfo.api_key_configured?"已配置":"未配置")+"</p>";
    if(health.diagnostics)$("diagnostic-results").textContent=JSON.stringify(health.diagnostics.payload,null,2);
}
$("check-environment").onclick=action(async()=>{await post("/api/system/check");notify("环境检查已启动，约数十秒后更新结果。");});
window.addEventListener("resize",()=>{curveChart?.resize();metricChart?.resize();confusionChart?.resize();stageChart?.resize();treeChart?.resize();});
setInterval(()=>{if(currentJob&&activeStates.includes(currentJob.status))action(pollJob)();if(currentPage==="experiments")action(loadExperiments)();},2500);
setInterval(()=>{action(loadHealth)();action(loadRecent)();if(currentPage==="tasks")action(loadTasks)();},10000);
action(async()=>{configInfo=await api("/api/config");resetMctsDefaults();$("review-category").innerHTML=configInfo.categories.map(c=>'<option value="'+c+'">'+esc(labels[c])+" / "+c+'</option>').join("");await Promise.all([loadHealth(),loadRecent()]);const previous=localStorage.getItem("vad-current-job");if(previous){try{await openJob(previous);}catch(e){localStorage.removeItem("vad-current-job");}}})();
