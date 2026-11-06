# V3 本地视频异常检测界面

从项目根目录运行，使用与 v3 检测相同的 Python / conda 环境：

```powershell
conda activate common
# 未安装 v3 检测环境时先执行：
python -m pip install -r vad_v3/requirement.txt
python -m pip install -r vad_v3/requirements-web.txt
$env:DASHSCOPE_API_KEY = "你的密钥"
python vad_v3/webapp/app.py
```

也可以将密钥写入 `vad_v3/webapp/.env`，此时无需每次执行上面的 `$env:` 设置：

```dotenv
DASHSCOPE_API_KEY=你的密钥
DASHSCOPE_MODEL=qwen3-vl-plus
```

`app.py` 启动时自动加载与自己同目录的 `.env`，不受启动工作目录影响，兼容带 BOM 的 UTF-8 文件。已有环境变量优先于 `.env`；修改文件后需要重启服务。通过界面启动的 `run.py` 子进程会继承该配置。直接运行检测 CLI 时仍需设置环境变量。

网页默认 VLM 为 `qwen3-vl-plus`，可通过 `DASHSCOPE_MODEL` 覆盖，并会将所用模型记录到任务文件。旧版 `qwen-vl-max` 已于 2026-10-10 下线（[官方公告](https://docs.qwencloud.com/changelog/model-deprecations/historical-mainline-models)）。网页显式传递 `--model`；历史检测 CLI 的默认值未改，直接调用时请指定可用模型，例如 `--model qwen3-vl-plus`。切换模型后，检测效果与历史实验不保证一致。

访问 <http://127.0.0.1:7860>。默认仅监听本机，使用 Waitress 提供 HTTP 服务。密钥不写入任务文件，也不发送给浏览器；`.env` 已加入 Git 忽略规则，可参考 `.env.example`。视频文件保存在本机；原有 v3 流水线会将抽取的帧发送给 DashScope。

## 使用

1. 选择或拖入本地视频，可以立即预览；未配置密钥也能手动播放任意区间。
2. 按需展开检测设置。网页默认使用 CUDA / NVIDIA GPU，无可用 NVIDIA GPU 时可手动切换到 CPU。MCTS `off` 和类别复核关闭沿用 v3 CLI 默认设置；选择 `refine` 才会应用通过验证的 MCTS 边界优化。
3. 点击「开始检测」。后台启动当前 Python 环境中的 `vad_v3/video_anomaly_detection/run.py`，显式传入配置的模型并保留其余默认参数。每次任务独立输出，避免与历史实验 manifest 冲突。同时只执行一个任务，运行期间可以查看日志。
4. 查看结果的异常类别、起止时间、置信度、原始描述。点「播放片段」或时间轴色块会跳到起始时间，并在结束时间暂停。勾选「循环片段」可重复播放。
5. 手动定位时支持 `12.5`、`01:20.5`、`00:01:20.5`。必须满足 `0 ≤ 开始 < 结束 ≤ 视频时长`。「完整视频」解除区间播放限制。
6. 导出 JSON，或通过「最近任务」重新打开结果；刷新页面会恢复最近查看的任务。
7. 点击最近任务右侧的「删除」，将该记录及视频、结果和日志移入本地回收目录，并从列表移除。删除当前查看的任务会清空播放器和结果区域；点击提示中的「撤销」可以恢复。正在上传或检测的任务不能删除。

## 原始描述的含义

v3 的最终边界可能经历 MCTS、合并、NMS 和后处理；首次扫描的区间不一定与最终边界相同。新增的描述字段保留首次 VLM 扫描的原文，不再次调用 VLM 改写：

```json
{
  "t_start": 12.4,
  "t_end": 20.8,
  "confidence": 0.91,
  "anomaly_type": "Fighting",
  "description": "Two people are fighting near the entrance.",
  "original_descriptions": [
    {
      "t_start": 12.0,
      "t_end": 18.0,
      "description": "Two people are fighting near the entrance."
    },
    {
      "t_start": 18.0,
      "t_end": 24.0,
      "description": "The confrontation continues."
    }
  ]
}
```

此 JSON 仅用于说明字段。合并多个候选时，`description` 为时间最早的一条原文，`original_descriptions` 保留全部来源及各自原始扫描区间。来源通过候选 ID 追溯，避免在边界移动后按时间重叠错误匹配。没有原文时不生成虚构描述；旧 JSON 无法凭空恢复已经丢失的描述，需要重新检测。

界面显示的是 v3 的最终类别。v3 现有后处理会统一同一视频内的异常类型，因此本系统没有引入“同一视频每个区间独立多类别分类”的新算法。原始描述是发现阶段的证据，并不是对精修后区间重新进行的描述。

## 存储与配置

默认写入 `vad_v3/web_data/<任务 ID>/`（已加入 Git 忽略规则）：

- `input.<扩展名>`：上传的视频。
- `job.json`：任务状态和选项。
- `result.json`：v3 原始输出及新增的描述字段。
- `run.log`：单视频检测日志。
- `metadata/result.manifest.json`：原有 v3 实验元数据。

可调整端口、存储位置、上传上限与检测超时：

```powershell
python vad_v3/webapp/app.py --port 7860 --data-dir "D:\vad_results" --max-upload-mb 2048 --timeout 3600
```

上传上限包含 multipart 请求开销。删除的任务保留在 `<数据目录>/.trash/<任务 ID>/`，方便撤销，因此不会立即释放磁盘空间。页面刷新后也可在停止服务时，将对应目录移回数据目录以恢复；如需永久释放空间，可自行清理回收目录。不要让多个服务实例同时使用同一存储目录。服务重启后，未完成任务会标记为中断，需要重新提交。

支持 MP4、WebM、MOV、M4V、AVI、MKV 上传，检测端使用 OpenCV 解码。浏览器能否播放还取决于视频编码，推荐 H.264 MP4 或 VP8/VP9 WebM；不能播放时界面会提示转码，未内置自动转码。播放器通过 HTTP Range 读取历史视频，支持拖动到指定时间。区间回看使用浏览器时间定位，结束暂停受帧率和浏览器调度影响，不是逐帧视频裁剪或片段文件导出。

## HTTP 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/config` | 密钥是否已配置、上传上限；不返回密钥 |
| POST | `/api/jobs` | multipart 上传 `video`，可带 `device`、`mcts_mode`、`context`、`full_classification` |
| GET | `/api/jobs` | 最近 30 个任务 |
| GET | `/api/jobs/<id>` | 状态、日志尾部，完成后包含 `result` |
| DELETE | `/api/jobs/<id>` | 将已结束的任务及文件移入本地回收目录 |
| POST | `/api/jobs/<id>/restore` | 从回收目录恢复任务 |
| GET | `/api/jobs/<id>/video` | 视频流，支持 Range / 206 |
| GET | `/api/jobs/<id>/result` | 下载完整结果 JSON |

## 验证

```powershell
conda run -n common python -m pytest vad_v3/video_anomaly_detection/tests vad_v3/webapp/tests -q
node --check vad_v3/webapp/static/app.js
```

测试覆盖原始描述在扫描合并、最终合并、NMS、MCTS 和配对尾流程中的传递；上传、子进程结果、失败/超时、重启恢复以及视频 Range 请求。模型计算在接口测试中使用替身，不消耗真实 API 配额。真实推理还需要有效密钥、原有模型权重及可用的检测环境。
