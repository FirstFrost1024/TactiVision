# TactiVision

面向 **SoccerTrack-V2 固定全景机位**的足球检测与逐帧二维位置重建。读取你手动下载的数据，不包含数据下载功能；推理要求提供本地 YOLO 权重，缺失时直接报错，不自动下载模型。

处理链路：原始全景视频 → 重叠切片足球检测 → 跨切片 NMS → 检测框中心 → 鱼眼点去畸变 → 逆单应变换 → 米制球场坐标 → 时序候选关联 → 逐帧 CSV / 可选预览视频。

## 数据来源与限制

- 数据源：[SoccerTrack-V2 官方数据集](https://huggingface.co/datasets/atomscott/soccertrack-v2)，数据许可 CC BY 4.0。请保留官方作者署名并按数据集要求取得访问权限。
- [官方代码与说明](https://github.com/AtomScott/SoccerTrack-v2)，[标定实现](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/calibration/generate_calibration_mappings.py)，[单应矩阵方向](https://github.com/AtomScott/SoccerTrack-v2/blob/main/scripts/calibration/calculate_homography.py)。本项目自行实现读取和反投影，不复制上游代码。
- 建议使用修正后的 **v1.2**，并记录实际下载的版本/提交。不同版本的时间对齐、132831 标定和 132877 同步有差异，不要混用标定与视频。已验证的 v1.2 提交为 `325bd4c57ee8e135ab652f3921d274526ada496b`。
- 官方 `ball/` 是由动作事件锚点进行直线、匀速插值的辅助轨迹，并非逐帧人工测量；不能直接当作逐帧足球检测框真值或据此声称检测精度。若用于弱监督，需另行构造和验证标签。`mot/` 的检测框真值只包含 22 名球员，不含足球，也不能作为足球检测标签。
- `mot/clips/<match>.mp4` 是挑战赛短片，其帧号属于短片本身；不能直接与全场视频、GSR 或 BAS 的帧号对齐。
- 单机位仅凭像素点和球场平面标定无法恢复腾空球的真实地面位置及高度。本项目输出检测中心对应视线与 **z=0 平面**的交点；贴地球也有球半径和标定带来的近似误差。
- 固定全景机位不需每帧重估标定。不能直接将原始鱼眼像素代入单应矩阵；也不能把 `mapx/mapy` 直接当作原图到去畸变图的正向点映射。

## 环境依赖与 Conda 安装

使用独立的 **`tactivision` Conda 环境**。项目中的安装、检查和运行命令均在该环境执行。双卡批处理按 Linux 环境配置，本机为两张 RTX 4080 SUPER，NVIDIA 驱动版本为 `580.178.04`。

环境依赖已写入根目录的 [`environment.yml`](environment.yml)：

- Python `3.11`、pip `>=24`：运行环境和 Python 包安装工具。
- PyTorch `2.7.0+cu128`、torchvision `0.22.0+cu128`：使用 CUDA 12.8 构建，沿用已通过本机双卡短测的组合。
- NumPy `2.4.6`：坐标及矩阵计算。
- OpenCV `opencv-python==4.11.0.86`：视频读写、鱼眼去畸变、投影和预览绘制。
- Ultralytics `8.4.198`：YOLO26 检测模型加载和推理。
- pytest `9.1.1`：自动化测试。
- Conda 的 `libgl`、`glib`：Linux OpenCV 所需的共享库。
- TactiVision 本地包：以 `.[detect,test]` 可编辑模式安装，其他间接依赖由包管理器解析。

PyTorch 使用官方 CUDA wheel 源，版本组合见 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v270)。CUDA 运行库随 PyTorch wheel 依赖安装，本项目推理无需另外安装 CUDA Toolkit 或 Conda 的 `cudatoolkit`。`huggingface-hub` 是可选的数据工具，本地重建及模型下载脚本都不需要它。

### 创建并激活环境

先安装 Miniconda 或 Anaconda，再在项目根目录执行：

```bash
cd /home/jerry/Demo/TactiVision
conda env create -f environment.yml
conda activate tactivision
```

`environment.yml` 中的 `-e .[detect,test]` 会同时安装项目和 `tactivision` 命令，无需再单独执行安装。此配置固定直接依赖版本，间接依赖未完全锁定。若终端提示无法执行 `conda activate`，先执行 `conda init bash`，重新打开终端后再激活环境。

每次打开新终端后使用：

```bash
conda activate tactivision
cd /home/jerry/Demo/TactiVision
```

OpenCV 统一使用 `opencv-python`，与 Ultralytics 的依赖保持一致；请在这个新环境中安装，避免混用 `opencv-python-headless` 等多个提供 `cv2` 的包。

### 检查依赖和双卡可用性

```bash
python -m pip check
python - <<'PY'
import sys
import cv2
import numpy as np
import torch
import torchvision
import ultralytics

print("Python:", sys.executable)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

本机双卡运行时，预期 Python 路径位于 Conda 的 `envs/tactivision/`，`CUDA available` 为 `True`，可见 GPU 数量为 `2`；如果设置了 `CUDA_VISIBLE_DEVICES`，数量只包含允许该进程使用的卡。后续可以先执行下方的双卡短测，再启动全量任务。`nvidia-smi` 顶部显示的是驱动支持的 CUDA 版本，PyTorch 实际使用的运行库版本以 `torch.version.cuda` 为准。

## 放置数据

当前完整数据位于 `datasets/SoccerTrack-v2/data/`，保留官方仓库和数据结构，比赛编号保持一致：

```text
datasets/
  SoccerTrack-v2/
    data/
      raw/117093/
        117093_camera_intrinsics.npz   # K、D、Knew；不读取 object 类型的外参数组
        117093_homography.npy          # 球场米制角点坐标 → 去畸变图像，3×3
        117093_keypoints.json          # "(x,y)": [u,v]，原始图像像素
      videos/117093/
        117093_panorama_1st_half.mp4
        117093_panorama_2nd_half.mp4
      mot/clips/                      # 挑战赛短片，与全半场视频分别使用自己的时间线
      gsr/                            # 比赛状态标注
      bas/                            # 球动作事件
      ball/                           # 辅助球轨迹
models/
  yolo26x.pt                          # 模型下载脚本的默认输出；需单独准备
```

`--dataset` 默认指向本项目中的 `datasets/SoccerTrack-v2/data/`，`--output` 默认指向本项目的 `outputs/`，均以项目位置解析，不依赖启动命令时的工作目录。显式传入的相对路径仍相对于当前工作目录；以下示例均在 TactiVision 项目根目录执行。

迁移数据时可用 `--dataset` 指定新的根目录；仍兼容 `datasets/raw/<match>/`、`datasets/soccertrack-v2/raw/<match>/` 和 `datasets/data/raw/<match>/`，也可用 `--calibration-dir` 直接指定标定目录。单视频命令通过 `--video` 显式指定输入；批处理命令只读取 `videos/<match>/<match>_panorama_{1st,2nd}_half.mp4`，避免混入挑战赛短片。无需修改下载的官方仓库配置即可使用本项目的 `tactivision` 命令。

通用 COCO 权重中的类别通常名为 `sports ball`，也支持 `ball`、`football`、`soccer ball` 自动识别。自训练模型可通过 `--class-id 0` 指定足球类别。通用权重仅作为基线，4K 全景中的小球通常需要专门训练的检测权重；代码不预设检测准确率。

## 模型选择与权重下载

精度优先建议先评估 **YOLO26x**；较低计算开销可选 **YOLO26l** 或 **YOLO26m**。官方 COCO 验证集上，三者在 640 输入下的 mAP50–95 分别为 57.5、55.0、53.1；这是通用目标指标，不能当作 SoccerTrack-V2 足球检测精度。[官方模型指标](https://docs.ultralytics.com/models/yolo26)

本场景建议保持重叠切片，先以 `imgsz=1024, tile_size=1024, overlap=0.2` 运行，再用相同片段的人工足球框比较召回率、误检和推理耗时。最后按比赛划分训练/验证/测试集做足球专项微调，避免相邻帧泄漏。哪个模型在该数据集上最好，需要这一步验证。

`scripts/download_models.py` 是显式执行的独立模型下载脚本，使用 Python 标准库。默认下载约 113 MiB 的 `yolo26x.pt` 到项目 `models/`，支持 YOLO26l、YOLO26m 和用于对照的 YOLO11x。推理命令本身仍只读取本地权重。

```bash
# 查看下载地址、大小、哈希与保存位置；不联网
python scripts/download_models.py --dry-run

# 下载默认 YOLO26x
python scripts/download_models.py

# 或下载较轻的版本进行对比
python scripts/download_models.py --models yolo26l yolo26m
```

脚本固定 [Ultralytics 官方 assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0)，按官方发布元数据校验文件大小和 SHA-256。已存在且校验通过则跳过；下载中断会清理临时文件，校验通过才写入正式 `.pt`。已有不匹配文件默认保留，可用 `--force` 在新文件成功下载后替换。连接失败时可重试，或手动从打印的官方地址获取权重。该脚本不下载 SoccerTrack-V2 数据。

下方重建命令使用下载脚本生成的 `models/yolo26x.pt`；如果准备了足球专项权重，将 `--weights` 改为其实际路径。YOLO26 需 `ultralytics>=8.4.0`，项目依赖已同步；COCO 的 `sports ball` 类别由程序自动识别。

## 先验证标定

```bash
tactivision check-calibration \
  --dataset datasets/SoccerTrack-v2/data --match 117093 \
  --output outputs/117093_calibration
```

此命令无需检测权重。`calibration_report.json` 保存 65 个关键点的逆投影残差、汇总统计及三个标定文件的 SHA-256。这些数值描述所提供关键点与当前 `K/D/Knew + H` 投影链的吻合程度，不是足球定位精度，也不能单独证明官方标定文件有问题或需要重新标定。原先使用的 2 m 提示阈值没有数据集验证依据，现已移除；残差数值继续保留。真实重建精度需结合原视频上的投影叠加和独立位置标注验证。程序直接使用官方标定，不自动重新拟合或修改数据集文件。

## 重建

```bash
tactivision run \
  --dataset datasets/SoccerTrack-v2/data --match 117093 \
  --video datasets/SoccerTrack-v2/data/videos/117093/117093_panorama_1st_half.mp4 \
  --weights models/yolo26x.pt \
  --device 0 --imgsz 1024 --tile-size 1024 --overlap 0.2 \
  --calibration-size 4096 1080 \
  --max-frames 250 --preview \
  --output outputs/117093_1st
```

首次运行可先检查 250 帧；处理全半场时去掉 `--max-frames 250`。无 GPU 改用 `--device cpu`。`4096 1080` 仅适用于这一比赛示例，应先核对每场原始视频的实际尺寸，不要套用于所有比赛。`--start-frame` 为当前输入视频中从零开始的帧号。

默认 `--space distorted`：在原始鱼眼全景图上检测，只对球心进行 `cv2.fisheye.undistortPoints(..., P=Knew)`，不需逐帧去畸变整个 4K 图像。

若输入已经去畸变，使用 `--space undistorted`，而且视频必须使用同一套 `Knew`、相同画布、无额外裁切或缩放。普通的裁切、缩放、播放器视角转换都需要相应更新标定；本版本要求输入与标定几何一致。`--calibration-size` 可强制尺寸检查，但相同尺寸本身不保证几何一致。

输出：

- `ball_positions.csv`：每个成功解码的输入帧一行，包括漏检帧。
- `calibration_report.json`：标定关键点残差及文件校验值。
- `summary.json`：输入路径、FPS、分辨率、检测覆盖率、参数、权重与标定哈希、坐标假设。覆盖率不是准确率。
- `reconstruction.mp4`：加 `--preview` 时生成，包含检测框、足球局部放大与球场位置；可直接用本地播放器查看。

## 双卡批量重建全部比赛

本机为两张 RTX 4080 SUPER（每卡约 16 GB）。`batch` 使用 `spawn` 启动两个独立进程，每张卡加载一个 YOLO 实例并持续复用；空闲进程领取下一段完整半场视频。标定和跟踪状态在每段视频开始时重新建立，视频之间不会串轨迹。该方式符合 [PyTorch CUDA 多进程要求](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html) 和 [Ultralytics 独立模型实例建议](https://docs.ultralytics.com/guides/yolo-thread-safe-inference)。

### 1. 进入项目并检查任务清单

后续命令均在项目根目录执行。已启动过任务时，可以直接跳到第 4 步查看进度。

```bash
cd /home/jerry/Demo/TactiVision
conda activate tactivision
```

确认 `models/yolo26x.pt` 已准备好，数据根目录为 `datasets/SoccerTrack-v2/data/`，然后检查清单：

```bash
# 列出全部任务，检查输入，不加载模型
python -m tactivision batch \
  --weights models/yolo26x.pt --devices 0 1 \
  --output outputs/all_matches --dry-run
```

当前完整数据应列出 10 场比赛、20 个半场，共 1,398,025 帧。`--dry-run` 只检查视频和标定路径、读取视频元数据，不执行检测，也不验证权重内容或 GPU 是否可用。

### 2. 可选：先做双卡短测

```bash
python -m tactivision batch \
  --weights models/yolo26x.pt --devices 0 1 \
  --matches 117093 --max-frames 10 \
  --output outputs/dual_gpu_smoke
```

这会处理 117093 两个半场的前 10 帧，分别保存到短测目录。控制台出现两张卡的 `model ready`、两个半场的 `completed`，且失败数量为 0，表示双卡调度和结果导出已跑通；检测精度需要另行验证。

### 3. 启动全部比赛

下面是两条完整的启动指令，**方式 A（前台）和方式 B（后台）任选一种执行**。两种方式都处理全部 10 场、20 个半场；每张 GPU 各运行一个模型实例，同时处理两个视频。同一输出目录只允许运行一个批处理。

#### 方式 A：前台运行——在当前终端执行

复制下面这条指令运行。它会占用当前终端，全局调度信息直接显示在终端中；请保持终端开启，按 Ctrl+C 可停止重建。各半场的帧进度仍写入各自的 `processing.log`，此指令不会自动创建全局 `batch.log`。

```bash
python -u -m tactivision batch \
  --weights models/yolo26x.pt --devices 0 1 \
  --imgsz 1024 --tile-size 1024 --overlap 0.2 \
  --output outputs/all_matches
```

#### 方式 B：后台运行——适合长时间处理

复制下面整个代码块（包括第一行 `mkdir` 和末尾的 `&`）运行。`nohup` 配合 `&` 将任务放到后台，终端会立即返回提示符，关闭终端后任务继续运行。全局日志追加写入 `outputs/all_matches/batch.log`，各半场帧进度写入各自的 `processing.log`；停止任务使用第 5 步的 `kill -TERM` 指令。

```bash
mkdir -p outputs/all_matches
nohup python -u -m tactivision batch \
  --weights models/yolo26x.pt --devices 0 1 \
  --imgsz 1024 --tile-size 1024 --overlap 0.2 --cpu-threads 24 --preview \
  --output outputs/all_matches \
  >> outputs/all_matches/batch.log 2>&1 < /dev/null &
```

上述前台、后台指令都不加 `--max-frames`，会处理所有 1,398,025 帧。各场视频的实际分辨率不同，批处理会分别读取尺寸；不要把 `4096×1080` 用于所有比赛。双卡提高任务吞吐量，每个视频仍按时间顺序处理。GPU 编号相对于启动时的 `CUDA_VISIBLE_DEVICES`，每个子进程仅暴露一张卡。

`--matches 117092 117093` 可指定部分比赛，`--cpu-threads 2` 控制每个进程的 CPU 计算线程。全量默认导出坐标与报告；需要渲染全部预览视频时加 `--preview`，会增加磁盘写入和编码耗时。

### 4. 查看进度和结果

按上面的后台命令启动后，全局调度日志保存在 `batch.log`。每个半场的逐帧进度写入自己的 `processing.log`，每 100 帧更新一次。

```bash
# 查看任务启动、半场完成或失败信息
tail -F outputs/all_matches/batch.log
```

按 Ctrl+C 退出日志查看后，可继续执行：

```bash
# 查看某个半场的帧处理进度；按需替换比赛编号和 1st/2nd
tail -F outputs/all_matches/117092/1st/processing.log
```

查看日志时按 Ctrl+C 只退出 `tail`，不会停止后台重建。以下命令汇总所有半场状态，并列出正在处理或失败的任务：

```bash
python - <<'PY'
from collections import Counter
import json
from pathlib import Path

state = json.loads(Path("outputs/all_matches/batch_status.json").read_text())
print("父进程 PID:", state["pid"])
print("任务状态:", dict(Counter(job["status"] for job in state["jobs"].values())))
for name, job in state["jobs"].items():
    if job["status"] in ("started", "failed"):
        print(name, job["status"], "GPU", job.get("device"), job.get("error", ""))
print("结束时间:", state.get("finished_at", "尚未正常结束"))
print("是否中断:", state.get("interrupted", False))
PY

# 查看两张 GPU 的使用情况
nvidia-smi
```

状态含义：`pending` 为排队，`started` 为已开始处理，`completed` 为本次完成，`skipped` 为复用已完成结果，`failed` 为该半场失败。若父进程已退出，结合 `interrupted`、`finished_at` 和日志判断结束原因，不能仅凭遗留的 `started` 状态判断仍在运行。

```text
outputs/all_matches/
  batch.log                       # 按后台命令启动时生成的全局日志
  batch_status.json               # 每个半场的排队、运行、完成或失败状态
  batch.lock                      # 防止两个批处理同时写入同一输出目录
  117093/
    1st/
      ball_positions.csv
      summary.json
      calibration_report.json
      processing.log
      completed.json              # 完整处理并校验输出后生成
    2nd/
      ...
```

### 5. 停止与中断后续跑

前台重建可按 Ctrl+C 停止。后台重建先从状态文件读取父进程 PID，并确认该 PID 的命令确实为 `tactivision batch`：

```bash
TACTIVISION_BATCH_PID=$(python -c 'import json; print(json.load(open("outputs/all_matches/batch_status.json"))["pid"])')
ps -p "$TACTIVISION_BATCH_PID" -o pid,etime,args
```

确认进程身份后停止：

```bash
kill -TERM "$TACTIVISION_BATCH_PID"
```

程序会停止两个子进程并保留已完成结果。等父进程退出后，重新执行第 3 步的方式 A（前台）或方式 B（后台）即可继续，保持推理参数和输出目录一致。

只有输入视频、标定、权重、推理参数、实现文件和输出哈希均匹配的完成任务才会跳过。**未完成或失败的半场从第一帧重新处理，不支持在半场内部接着上次的帧号继续。** 修改上述输入、参数或实现文件会使相关完成结果重新计算；短测结果不会被当作整半场完成。程序会核对解码帧数，提前解码中断时不会写入完成标记。单个半场失败后继续其他视频，并在批处理结束时返回非零退出码；失败原因见该半场的 `processing.log`。

状态文件中的时间采用 UTC。

### 6. 手动启用每日 08:30–16:30 双卡任务

定时脚本为 `scripts/schedule_batch.py`，**只在你手动运行后生效**，不会注册 cron、系统服务或开机自启。默认按北京时间 `Asia/Shanghai` 每天（含周末）08:30 启动，16:30 向批处理父进程发送 SIGTERM，等待其清理双卡子进程；超过 60 秒仍未退出会强制清理本次批处理的整个进程组。窗口判断约每秒一次。第二天 08:30 使用相同参数和目录重新运行。

脚本已包含比赛用球选择修正：`--pitch-region keypoints --focus-pitch --outside-static-seconds 1.0 --off-field-grace 1.0 --save-candidates`，默认双卡 `0 1`、512 切片、1024 推理尺寸，输出为 `outputs/all_matches_v2`。无需额外打开“选择修正”开关。已完成且核验匹配的半场会跳过；**16:30 被中断的半场次日从第一帧重算，不支持帧级续跑**。当天提前完成或报错退出后，脚本等待次日，不在同一天反复重启。

以下命令在项目根目录执行。本机当前 `.venv/bin/python` 已有 PyTorch 和 Ultralytics，使用它可避免误用其他 Conda 环境；如果使用自行配置的 `tactivision` 环境，将 `.venv/bin/python` 换为该环境的 Python。批处理自动使用启动定时脚本的同一个 Python。

```bash
cd /home/jerry/Demo/TactiVision

# 只检查时间设置和将要执行的命令，不启动任务、不写结果
.venv/bin/python scripts/schedule_batch.py --dry-run
```

准备好后，**手动执行下面整个代码块才会启用定时任务**：

```bash
cd /home/jerry/Demo/TactiVision
mkdir -p outputs/all_matches_v2
nohup .venv/bin/python -u scripts/schedule_batch.py \
  >> outputs/all_matches_v2/schedule.log 2>&1 < /dev/null &
```

08:30 前启动会等待当天 08:30；08:30（含）至 16:30（不含）启动会立即处理；16:30 后启动会等待次日。关闭终端后定时脚本继续运行，电脑重启后需手动重新启用。请保持电脑开机且不休眠。不要同时启动指向该输出目录的前台/后台批处理；脚本的 `schedule.lock` 防止重复调度，批处理的 `batch.lock` 防止并发写入结果。

```bash
# 查看定时启停记录，按 Ctrl+C 仅退出日志查看
tail -F outputs/all_matches_v2/schedule.log

# 查看双卡重建进度
tail -F outputs/all_matches_v2/batch.log
```

手动关闭定时任务（也会停止它正在运行的双卡批处理）：

```bash
TACTIVISION_SCHEDULE_PID=$(cat outputs/all_matches_v2/schedule.pid)
ps -p "$TACTIVISION_SCHEDULE_PID" -o pid,etime,args
# 确认上面的命令为 scripts/schedule_batch.py 后执行
kill -TERM "$TACTIVISION_SCHEDULE_PID"
```

停止定时脚本后不会在第二天自动重启；再次执行 `nohup` 代码块即可重新启用。只停止 batch 父进程会保留定时脚本，它仍将在次日启动。可通过 `--start 08:30 --stop 16:30 --timezone Asia/Shanghai` 调整窗口，`--cpu-threads 24` 调整每卡 CPU 线程，`--preview` 额外生成预览；重新启用时保持参数一致，避免完成标记失配而重算。

## 比赛用球选择修正与候选缓存

新版默认使用 `--pitch-region keypoints --focus-pitch --save-candidates`：

- 按官方关键点在输入图像中构成比赛区域多边形，保留鱼眼画面弯曲边线。直接读取官方文件，不重新拟合或修改标定。
- 新关联段从场内候选开始，避免高置信度场边备用球抢占目标。已建立轨迹可以在通过距离门限的条件下，出界后继续最多 `--off-field-grace 1.0` 秒；界外坐标保留原值。
- 持续在场外 4 px 邻域中静止至少 `--outside-static-seconds 1.0` 秒的候选被抑制；场内静止球不受此规则影响。短时丢失仍然输出空坐标，长时间出界和跨出区域轮廓的高球可能漏记。
- 默认在比赛区域包围框及其 32 px 边缘内切片检测，框坐标转换回原视频像素。可用 `--no-focus-pitch` 恢复整幅全景检测。

小球较远时可采用 `--tile-size 512 --imgsz 1024`，让小球在网络输入中放大；相较 1024 切片会增加计算量。验证和重建新版结果时使用新的输出目录，方便与旧结果对照：

```bash
# 先检查一段已知包含比赛用球的片段，生成检测预览和候选缓存
python -m tactivision run \
  --match 117093 \
  --video datasets/SoccerTrack-v2/data/videos/117093/117093_panorama_1st_half.mp4 \
  --weights models/yolo26x.pt --device 0 \
  --start-frame 33800 --max-frames 200 \
  --tile-size 512 --imgsz 1024 --preview \
  --output outputs/selection_validation/117093_v2

# 短片抽检通过后，双卡重建全部比赛
python -m tactivision batch \
  --weights models/yolo26x.pt --devices 0 1 \
  --tile-size 512 --imgsz 1024 --overlap 0.2 \
  --output outputs/all_matches_v2
```

`--pitch-region none --no-focus-pitch` 可用于无区域约束的对照实验。区域约束和候选筛选只能改善目标选择；检测器未检出比赛用球时，仍需改进检测模型或训练足球专项权重，不能由备用球补齐轨迹。场内比例和覆盖率都不能作为真实检测准确率。

默认在每个结果目录保存 `candidates.jsonl.gz` 和 `candidates_meta.json`，记录**当前检测设置下的全部原始候选框**，在比赛用球筛选之前写入；完成请求帧数后发布，并记录源视频、帧范围、检测参数、检测区域和校验值。可用 `--no-save-candidates` 关闭。旧版结果仅保存被选中的一个目标，无法从旧 CSV 恢复未保存的候选，需重新检测一次才能生成缓存。

之后调整关联规则，可直接复用同一个短片缓存：

```bash
python -m tactivision run \
  --match 117093 \
  --video datasets/SoccerTrack-v2/data/videos/117093/117093_panorama_1st_half.mp4 \
  --weights models/yolo26x.pt --device cpu \
  --candidate-cache outputs/selection_validation/117093_v2 \
  --start-frame 33800 --max-frames 200 \
  --off-field-grace 1.0 --outside-static-seconds 1.0 --preview \
  --output outputs/selection_validation/117093_replay
```

缓存复关联会解码视频并在 CPU 上执行关联、投影和可选预览，不再次加载或运行 YOLO。缓存不能补回低于原检测阈值、原检测区域之外或原帧范围之外的候选；输入视频、权重和图像坐标空间必须匹配。修改 `--imgsz` 或 `--tile-size` 不能改变缓存里的已有检测结果，摘要会保留原检测来源。完整比赛缓存从第 0 帧开始时，可以省略 `--start-frame` 和 `--max-frames`。

新版全量完成后生成对应的质量报告：

```bash
python -m tactivision review \
  --input outputs/all_matches_v2 --output outputs/review_v2 \
  --samples 4 --clip-seconds 4
```

新增 CSV 列 `playing_candidate_count` 为该帧原始候选中落在图像比赛区域内的数量，`selected_in_playing_region` 表示被选中的像素是否位于该区域；它与 `in_pitch`（米制投影是否位于球场矩形内）分别记录。

## 重建完成后的轨迹可视化与质量检查

在 Conda 环境中执行以下命令，直接读取已保存的结果，无需再次加载 YOLO 或运行 GPU 推理：

```bash
conda activate tactivision
python -m tactivision review \
  --input outputs/all_matches \
  --output outputs/review \
  --samples 4 --clip-seconds 4
```

每个半场会核对完成标记中的输出 SHA-256、CSV 行数、连续帧号、时间戳、检测值和汇总计数，统计检测覆盖率、投影位置的界外比例、像素静止段及缺失段。原始 CSV 和官方标定保持原样，轨迹不插值、不平滑。短片解码和绘制使用 CPU；本机有 FFmpeg 时转为 H.264 方便浏览器播放，否则保留 OpenCV 的 MP4 编码。`--clip-seconds 0` 可只生成图像。

```text
outputs/review/
  index.html                       # 可在本机浏览器打开的离线交互报告
  quality_report.json              # 全部半场的核验结果和统计
  quality_summary.csv              # 每个半场一行的汇总
  117093/1st/
    contact_sheet.jpg              # 多个时刻的原视频、选中目标及局部放大
    trajectory.svg                 # 球场米制坐标图（最多抽样 1000 点）
    review.mp4                     # 最长像素静止段附近的抽检短片
    metrics.json
```

报告可以直接打开 `outputs/review/index.html`，也可以用本机 HTTP 服务查看：

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory outputs/review
# 浏览器打开 http://127.0.0.1:8765/
```

**检测覆盖率不等于比赛用球跟踪准确率。** 通用 `sports ball` 检测器也会检测场边备用球；简单的置信度和距离关联可能长期锁定静止球。`stationary_target_30s` 表示同轨连续像素中心距段起点不超过 2 px 且持续至少 30 秒，`mostly_outside_pitch` 表示至少一半已选位置投影到球场外，`dominant_4px_cell` 表示至少一半已选像素落在同一个 4×4 px 格子，`gap_2s` 表示缺失至少 2 秒。这些标记用于安排复查，不能直接当作误检标签。

本次全量检查已在抽检画面中观察到部分半场长期选中场边静止球。应先结合抽检图和视频核实目标选择，再决定是否增加比赛区域约束、静态候选抑制或比赛用球专项模型。没有独立人工足球标注时，不报告精确率、召回率和真实米制定位精度；辅助 `ball/` 插值轨迹不作为逐帧检测真值。

## 坐标与状态

`frame` 从 0 开始，`timestamp_ms = frame / 视频FPS × 1000`，相对于传入视频的起点。当前按恒定帧率视频处理，符合 SoccerTrack-V2 的使用场景。

`u_px, v_px` 为原输入图像中的框中心，单位像素。`undistorted_u_px, undistorted_v_px` 为 `Knew` 对应画布中的点。

`x_m, y_m` 保留官方标定关键点的角点原点坐标：x 沿球场长度、y 沿宽度向近侧边线增大。常规范围为 `[0,105] × [0,68]`。另提供 `x_center_m = x_m - L/2`、`y_center_m = W/2 - y_m`，以中心为原点、y 向远侧边线增大。该中心坐标公式显式定义，比较其他标注前需核对其实际轴向。

`status` 有三种：`detected`（选中候选并投影）、`missing`（没有检测候选）、`rejected`（有候选但投影无效或时序门限不通过）。后两种的坐标、框与置信度为空。`track_id` 是局部关联段编号，超出 `--reset-after 0.4` 秒后重新关联时递增，不保证遮挡前后身份。

多个候选先按米制距离门限关联：`允许距离 = max_speed × 时间差 + gate_slack`。默认 `max_speed=45 m/s`、`gate_slack=2 m`；标定误差较大或球腾空时会误拒绝，需要结合预览调整。它是候选筛选基线，不是完整多假设追踪器。

漏检不插值、不生成伪检测；界外坐标不截断，使用 `in_pitch` 标记。`confidence` 为检测器分数，不代表坐标精度或物理位置正确的概率。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

关闭 pytest 第三方插件自动加载，避免当前主机的 ROS 插件干扰。测试覆盖鱼眼点去畸变与逆单应链路、矩阵方向、坐标原点和轴向、切片坐标还原、重复检测去重、异常跳变与遮挡后重关联、逐帧 CSV 与时间对齐、界外球保留、视频尺寸检查和预览编码。测试使用合成几何和受控检测结果，无需下载数据或模型；真实足球检测精度需在你提供的视频和人工标注上单独验证。
