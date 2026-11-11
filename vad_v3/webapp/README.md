# V3 视频异常检测与时序定位辅助分析系统

本地单用户系统：视频上传与复用 → 持久化排队 → V3 单视频检测 → 区间回看 → 过程分析 → 人工复核 → 导出与实验评价。新任务默认设备为 CUDA，MCTS 默认开启精修（refine），仅应用通过验证的边界；仍可手动关闭或选择观察模式。搜索按候选触发条件和预算执行，可能增加 VLM 调用与耗时。历史任务保留原配置；算法命令行的基线默认值不受此网页设置影响。**VLM 描述保留发现阶段原文，不要求中文，也不额外调用模型翻译。**

## 1. 启动、停止与安装

在项目根目录的 PowerShell 中执行：

~~~powershell
# 同时启动 Web 和独立 Worker，后台运行
.\vad_v3\webapp\scripts\start.ps1
# 查看状态
.\vad_v3\webapp\scripts\status.ps1
# 停止两个服务及当前推理
.\vad_v3\webapp\scripts\stop.ps1
# 修改 .env 或更新代码后重启
.\vad_v3\webapp\scripts\restart.ps1
~~~

访问 [http://127.0.0.1:7860](http://127.0.0.1:7860)。关闭浏览器不会停止服务。脚本优先使用当前用户的 .conda/envs/common/python.exe，没有时使用 PATH 中的 Python；可用 -PythonExe、-DataDir、-Port 指定解释器、数据目录和端口。

等价命令（也可用于不允许执行 PowerShell 脚本的环境）：

~~~powershell
conda activate common
python -m vad_v3.webapp.service start
python -m vad_v3.webapp.service status
python -m vad_v3.webapp.service stop
python -m vad_v3.webapp.service check
~~~

首次在新环境安装：

~~~powershell
# 已有算法环境，只补 Web 依赖
.\vad_v3\webapp\scripts\install.ps1
# 新环境补全部 Python 依赖；CUDA 版 PyTorch 需与目标机器兼容
.\vad_v3\webapp\scripts\install.ps1 -Full
~~~

当前机器使用 C:/Users/Gigabyte/.conda/envs/common/python.exe。新增依赖 psutil、imageio-ffmpeg 已安装，后者附带 FFmpeg；ECharts 5.6.0 及许可证保存在 static/vendor，网页无需访问 CDN。CLIP、VideoMAE 权重已存在于本机 Hugging Face 缓存，MIL 使用算法目录中的原权重。包版本见 environment-2026-11-11.json。

2026-11-11 已验证本机 PyTorch 2.7.1+cu128 和 RTX 5060 Ti CUDA 可用，之前的 WinError 4551 不再复现。当前环境状态以系统诊断和实际推理为准。

## 2. 密钥与配置

Web 和 Worker 都从 vad_v3/webapp/.env 加载配置，支持 UTF-8 BOM，不依赖启动目录：

~~~dotenv
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_MODEL=qwen3-vl-plus
~~~

已有环境变量优先。修改后重启两个进程。接口只返回密钥是否配置，运行快照不含密钥。原录像保存在本机，检测时选中的图像会发送给 DashScope。

任务冻结完整算法配置、MCTS 预算、后处理开关、场景、VLM 请求参数、源码内容哈希、依赖版本和缓存模型修订标识。排队后源码或运行环境改变，任务会提示重新创建。历史导入不能倒推出原执行环境。

Web 新任务默认 `mcts_mode=refine`、`acceptance_policy=force_best`。页面研究参数可选择接收策略；显式传入的策略优先。打开或重试旧任务仍使用旧配置，旧目录导入保留原 strict 接收规则。四组配对实验的策略分别由实验配置指定。

## 3. 页面与操作

| 页面 | 功能 |
| --- | --- |
| 分析工作台 | 预览、上传、设备与研究参数、排队、区间与原始描述、报告导出 |
| 视频库 | 搜索、分页、复用、删除与恢复 |
| 任务中心 | 筛选、分页、取消、重试、日志、软删除与恢复 |
| 过程与复核 | 类别依据、曲线、实际 VLM JPEG、阶段边界、Q-score 分量、MCTS 节点和评价、独立修订 |
| 实验中心 | 历史/JSON 导入、从任务组建对比、离线评价、混淆矩阵、错误案例、指标与匹配审计 |
| 系统状态 | Web/Worker 健康、Python/包版本、CUDA、权重缓存、FFmpeg、密钥状态 |

点击区间会跳到起点、在终点暂停，支持循环。手动输入支持 12.5、01:20.5、00:01:20.5。浏览器定位受编码与调度影响，不是逐帧裁剪；浏览器时长可能与 OpenCV 读取值稍有不同，检测/评价保存自己的源时间轴。推荐 H.264 MP4；自动转码属于 P2。

最终区间可关联多个 original_descriptions，各自描述与时间属于最初发现区间，未按精修后的边界重写。类别沿用 V3 视频内统一策略，不宣称增加区间独立多类别分类。

评分用于排序，未校准为概率。空检测表示本流程未检出异常，不是“已证明安全”。API、响应格式、DLL 错误和超时记录为失败，不作为 Normal 结果。

## 4. 复核、证据与导出

支持确认、误报、修正类别/边界和补充漏检。修订使用独立 ID、递增版本、并发冲突检查，不覆盖原 result.json，也不自动变成测试 GT。

新任务在真实调用位置记录实际输入 JPEG 字节、提示、响应、范围和阶段快照。重复画面无法唯一恢复时间时明确标记，不猜测证据。旧任务缺少数据时显示“未记录”。

工作台导出模型 JSON、含独立复核投影的 CSV、HTML 报告；HTML 可打印为 PDF。曲线导出 PNG。实验可下载指标 JSON/CSV 和逐预测匹配审计。

记录实际请求/token、执行耗时和峰值 RAM；阶段耗时按已有轨迹展示。未知用量不按零处理。当前没有可靠的逐进程 GPU 峰值采集，显示“未记录”；未配置价格时不估算费用。配对分支共享上游，总用量不能解释为分支独立成本。

## 5. 实验与评价协议

完整小型实验包见 [examples/experiment.json](examples/experiment.json)，格式如下：

~~~json
{
  "name": "开发集比较",
  "ground_truth": {
    "video_a": {"category": "Fighting", "fps": 30.0, "frame_count": 300, "segments": [[60,120]]}
  },
  "groups": {
    "baseline": {
      "video_a": {
        "duration": 10.0, "is_normal": false,
        "anomalies": [{"t_start":2.0,"t_end":4.0,"confidence":0.8,"anomaly_type":"Fighting"}],
        "timestamps": [0.0,2.0,4.0,9.0], "frame_scores": [0.1,0.9,0.8,0.1]
      }
    }
  }
}
~~~

GT 是原视频 **0 起始、右端不包含** 的整数帧区间。闭区间数据须按来源规范转换并记录来源，不猜测。Normal 区间为空。各组必须覆盖同一完整视频集合，缺失预测则拒绝评价。GT 越界项列入清单；AP 保留原标注，帧级评价截到实际视频范围。

“从已完成任务创建对比实验”只需上传以视频文件名（去扩展名）为键的 GT，两组各选相同视频的一个已完成任务：

~~~json
{"Explosion001_x264": {"category":"Explosion","segments":[[60,120]]}}
~~~

系统检查同名视频哈希，并读取上传时的 FPS/帧数。任务、attempt、视频、代码和配置哈希保存在实验来源中。无 GT 的普通视频不显示准确率。

| 协议 | 定位 AP | 帧级 ROC-AUC/AUPR |
| --- | --- | --- |
| vadclip_native_fps | 每次打开原视频读取 FPS，13 类，tIoU 0.1–0.5，rank AP；异常子集和全视频分别汇总 | 重新读取原始帧数，半开 GT，float64，线性插值、首尾保持 |
| report_historical_v1 | 固定 30 FPS，其余 rank AP 规则一致 | int(duration*30) 帧，闭合 GT 终点、float32，复现历史取整及精度 |

AUPR 是 PR 曲线梯形积分，average_precision 另存。缺失曲线时不计算汇总 AUC 且列出视频，不补零或伪造曲线。单类帧标签的 AUC 无定义。图表采样不参与评价。

历史导入读取仓库 experiments/v217/results、本机原视频及 F:/dataset/ucf_crime/data_parquet/test-00000-of-00001.parquet，不读取 FPS 清单。本机数据集位置可通过 VAD_DATASET_DIR 环境变量配置后重启服务。原生 FPS 评价要求原视频可读，缺失时直接失败；JSON 包中的 FPS/帧数只保留为输入记录，计算使用本次探测值。JSON 导入可通过预测中的 video_path 指向数据集或系统视频目录，省略时按视频标识在数据集中查找。显式历史固定 30 FPS 协议可在没有原视频时使用。

## 6. 存储、恢复与备份

~~~text
web_data/
  system.sqlite3
  videos/<video_id>/input.mp4
  jobs/<job_id>/attempts/<attempt_id>/
    detector_config.json  manifest.json
    run.log               result.json
    research.json         curves.json
    stage_events.jsonl    result.normalized.json
    evidence/*.jpg
  experiments/<experiment_id>/
    input.json
    evaluations/<evaluation_id>/metrics.json
                                 matching_audit.json
~~~

SQLite 使用 WAL 和短事务。Worker 跨数据目录共用推理锁，单 GPU 串行执行。取消/超时先确认进程树停止，再提交终态；重试新建 attempt，保留旧日志。恢复时核对 PID 创建时间和执行目录，停止遗留进程，把旧执行标为中断，不自动重跑付费任务。

任务和视频软删除可恢复，不立即释放磁盘。删除任务不删除共享视频。启动时幂等导入旧任务目录和 .trash，原目录不变。

停止服务后备份；恢复必须指定尚不存在的目录：

~~~powershell
python -m vad_v3.webapp.service stop
python -m vad_v3.webapp.service backup --output "D:\vad-backups\system.zip"
python -m vad_v3.webapp.service restore --output "D:\vad-backups\system.zip" --data-dir "D:\vad-restored"
python -m vad_v3.webapp.service start --data-dir "D:\vad-restored"
~~~

备份包括数据库、视频、实验，不含外部 .env 和模型缓存。备份取得数据锁，恢复检查 ZIP 路径并拒绝覆盖。服务日志为 web.service.log、worker.service.log；推理日志在各 attempt 中。

## 7. HTTP 接口

| 方法 | 路径 | 功能 |
| --- | --- | --- |
| GET | /api/config、/api/health | 默认配置、健康状态 |
| POST | /api/system/check | 异步环境检查 |
| GET / POST | /api/videos | 分页列表 / multipart 上传 |
| GET | /api/videos/ID/stream | Range 视频流 |
| DELETE / POST | /api/videos/ID、/api/videos/ID/restore | 删除 / 恢复 |
| GET / POST | /api/jobs | 分页列表 / 提交 video_id 和 config；支持 Idempotency-Key |
| GET | /api/jobs/ID | 状态、attempt、结果、日志尾部 |
| POST | /api/jobs/ID/cancel、/retry | 取消 / 重试 |
| DELETE / POST | /api/jobs/ID、/api/jobs/ID/restore | 删除 / 恢复 |
| GET | /api/jobs/ID/analysis、/curves、/events、/mcts | 研究数据 |
| GET | /api/jobs/ID/evidence/FILE | 已登记的证据图 |
| POST | /api/jobs/ID/reviews | expected_revision 并发检查的修订 |
| GET | /api/jobs/ID/result、/export?format=csv 或 html | 原结果与报告 |
| GET / POST | /api/experiments | 实验列表 / JSON 包导入 |
| POST | /api/experiments/import-history、/from-jobs | 历史导入 / 任务对比 |
| POST | /api/experiments/ID/evaluate | 指定协议开始评价 |
| GET | /api/evaluations/ID、/download | 状态与指标；下载支持 format=csv、audit=1 |

仅监听 127.0.0.1，未提供多用户认证或公网部署。修改请求检查 Origin；文件路径由服务端解析，证据必须已登记。

## 8. 验证与范围

~~~powershell
python -m pytest vad_v3/webapp/tests tests/test_native_fps_ap.py tests/test_llm_usage.py -q
python -m pytest vad_v3/video_anomaly_detection/tests -q
node --check vad_v3/webapp/static/app.js
~~~

Web 验收使用可控子进程和独立临时目录，不消耗 API；原生 FPS 评估测试使用临时生成的真实视频。算法的真实推理结果另见 experiments 中各轮记录。

本次实现单机 P0/P1 核心工程。P2 多用户、实时流、分布式、批量推理、自动转码不在本次范围。算法精度收益仍须独立受控实验，不能从界面功能推导。
