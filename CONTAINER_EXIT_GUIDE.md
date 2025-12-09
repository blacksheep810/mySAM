# 退出容器后继续运行训练指南

## ?? 重要说明

**关键问题**：如果容器被停止或重启，容器内的所有进程都会终止，包括使用 nohup/setsid 启动的进程。

**解决方案**：
1. **容器内**：使用 `setsid` + `nohup` 让进程脱离终端
2. **容器外**：使用 Docker detach 模式（最可靠）
3. **持久化**：确保容器不会停止，或使用 Docker 的 restart 策略

## ? 统一训练脚本使用方法

所有训练功能都集成在 `train.sh` 脚本中，支持多种运行模式和管理功能。

### 基本命令

```bash
# 启动训练（默认守护进程模式）
bash train.sh start

# 查看训练状态
bash train.sh status

# 实时查看日志
bash train.sh follow

# 停止训练
bash train.sh stop

# 查看帮助
bash train.sh help
```

## ? 启动训练（start 命令）

### 运行模式

`train.sh` 支持多种运行模式：

| 模式 | 说明 | 退出终端 | 退出容器 | 适用场景 |
|------|------|----------|----------|----------|
| **daemon**（默认） | 守护进程模式，使用 setsid | ? | ? | 容器内运行，退出终端后继续 |
| **screen** | Screen 会话模式 | ? | ? | 需要随时重新连接查看 |
| **tmux** | Tmux 会话模式 | ? | ? | 需要多窗口管理 |
| **foreground** | 前台运行模式 | ? | ? | 调试和测试 |

### 基本用法

```bash
# 1. 默认守护进程模式（推荐）
bash train.sh start

# 2. 使用 screen 模式
bash train.sh start --mode screen

# 3. 使用 tmux 模式
bash train.sh start --mode tmux

# 4. 前台运行（调试用）
bash train.sh start --mode foreground
```

### 自定义参数

```bash
# 设置批次大小和启用混合精度
bash train.sh start --batch-size 1 --use-amp

# 启用梯度检查点（节省显存）
bash train.sh start --use-gc

# 设置梯度累积步数
bash train.sh start --grad-accum 4

# 只解冻最后1层
bash train.sh start --unfreeze-k 1

# 设置训练轮数
bash train.sh start --epochs 50

# 组合使用多个参数
bash train.sh start --batch-size 1 --use-amp --use-gc --grad-accum 4
```

### 完整参数说明

```bash
bash train.sh start [选项]

选项:
  --mode MODE          运行模式: daemon(默认), screen, tmux, foreground
  --batch-size N       批次大小（默认: 2）
  --use-amp            启用混合精度训练
  --use-gc             启用梯度检查点
  --grad-accum N        梯度累积步数（默认: 2）
  --unfreeze-k N        解冻最后K层（默认: 2）
  --epochs N           训练轮数（默认: 30）
```

## ? 管理命令

### 查看训练状态

```bash
bash train.sh status
```

显示信息：
- 进程/会话运行状态
- PID 或会话名
- GPU 使用情况
- 最新日志片段

### 查看日志

```bash
# 查看最新日志（最后50行）
bash train.sh logs

# 实时跟踪日志
bash train.sh follow
```

### 停止训练

```bash
bash train.sh stop
```

会自动停止所有运行中的训练进程或会话。

### 列出所有日志

```bash
bash train.sh list
```

显示所有训练日志文件及其信息。

### 清理旧日志

```bash
# 清理7天前的日志文件
bash train.sh clean
```

## ? 方法对比

| 方法 | 退出终端 | 退出容器 | 容器重启 | 可靠性 |
|------|----------|----------|----------|--------|
| daemon (setsid) | ? | ? | ? | ??? |
| screen | ? | ? | ? | ??? |
| tmux | ? | ? | ? | ??? |
| Docker detach | ? | ? | ? | ???? |
| Docker restart=always | ? | ? | ? | ????? |

## ? 方法1：守护进程模式（daemon，推荐）

### 启动训练

```bash
bash train.sh start
# 或显式指定
bash train.sh start --mode daemon
```

### 原理
- 使用 `setsid` 创建新会话，完全脱离终端
- 即使退出当前 shell，进程也会继续运行
- **但容器停止/重启仍会终止进程**

### 验证

```bash
# 启动训练后，退出 shell
exit

# 重新进入容器，检查进程
bash train.sh status

# 或查看日志
bash train.sh follow
```

## ? 方法2：Screen/Tmux 模式（可重新连接）

### Screen 模式

```bash
# 启动训练
bash train.sh start --mode screen

# 重新连接会话
screen -ls                    # 列出所有会话
screen -r sam_train_*         # 连接到会话

# 断开连接（不停止训练）：按 Ctrl+A，然后按 D
# 停止训练：在会话中按 Ctrl+C
```

### Tmux 模式

```bash
# 启动训练
bash train.sh start --mode tmux

# 重新连接会话
tmux ls                        # 列出所有会话
tmux attach -t sam_train_*     # 连接到会话

# 断开连接（不停止训练）：按 Ctrl+B，然后按 D
# 停止训练：在会话中按 Ctrl+C
```

## ? 方法3：Docker Detach 模式（最可靠）

### 在宿主机上运行（推荐）

```bash
# 方法1：使用 docker exec -d
docker exec -d <container_name> bash /root/workspace/mySAM/train.sh start

# 方法2：自定义参数
docker exec -d <container_name> bash -c "
  cd /root/workspace/mySAM && 
  bash train.sh start --batch-size 1 --use-amp
"
```

### 在容器内运行

```bash
# 1. 启动训练
bash train.sh start

# 2. 使用 Ctrl+P, Ctrl+Q 退出容器（不停止容器）
# 或使用 exit，但确保容器不会停止
```

### 重新连接查看

```bash
# 重新进入容器
docker exec -it <container_name> bash

# 查看训练状态
bash train.sh status

# 查看日志
bash train.sh follow
```

## ? 方法4：Docker Restart 策略（最持久）

### 创建带 restart 策略的容器

```bash
docker run -d \
  --name sam_train \
  --restart=unless-stopped \
  --gpus all \
  -v /path/to/data:/root/workspace/mySAM/data \
  -v /path/to/checkpoints:/root/workspace/mySAM/checkpoints \
  -v /path/to/outputs:/root/workspace/mySAM/outputs \
  your_image:tag \
  bash -c "cd /root/workspace/mySAM && bash train.sh start"
```

### 或修改现有容器

```bash
# 更新容器的 restart 策略
docker update --restart=unless-stopped <container_name>
```

## ? 验证进程是否在运行

### 方法1：使用脚本命令

```bash
# 查看状态（推荐）
bash train.sh status
```

### 方法2：手动检查

```bash
# 检查进程
ps aux | grep model.py

# 或使用 PID 文件
ps -p $(cat logs/train_*.pid)
```

### 方法3：检查 GPU 使用

```bash
nvidia-smi
```

### 方法4：查看日志

```bash
# 实时查看
bash train.sh follow

# 查看最后几行
bash train.sh logs
```

## ?? 完整工作流程

### 场景1：容器内启动，退出容器后继续运行

```bash
# 1. 进入容器
docker exec -it <container_name> bash

# 2. 启动守护进程训练
cd /root/workspace/mySAM
bash train.sh start

# 3. 验证进程运行
bash train.sh status

# 4. 退出容器（使用 Ctrl+P, Ctrl+Q 或 exit）
# 注意：确保容器不会停止

# 5. 重新进入容器查看
docker exec -it <container_name> bash
bash train.sh status
```

### 场景2：从宿主机直接启动（最可靠）

```bash
# 在宿主机上运行
docker exec -d <container_name> bash -c "
  cd /root/workspace/mySAM && 
  bash train.sh start --batch-size 2 --use-amp
"

# 查看日志
docker exec <container_name> bash /root/workspace/mySAM/train.sh follow
```

### 场景3：使用 Screen 模式，随时查看进度

```bash
# 1. 启动训练（screen 模式）
bash train.sh start --mode screen

# 2. 记录会话名（例如：sam_train_20240101_120000）

# 3. 退出 shell
exit

# 4. 重新进入容器，连接会话
docker exec -it <container_name> bash
screen -r sam_train_20240101_120000
```

## ?? 注意事项

1. **容器停止 = 进程终止**
   - 如果容器被停止（`docker stop`），所有进程都会终止
   - 使用 `docker restart=unless-stopped` 可以自动重启容器

2. **容器重启 = 进程终止**
   - 如果容器重启，需要重新启动训练
   - 使用 Docker restart 策略可以自动重启容器和训练

3. **数据持久化**
   - 确保训练数据和输出目录已挂载到宿主机
   - 使用 `-v` 参数挂载卷

4. **日志持久化**
   - 日志文件保存在 `./logs/` 目录
   - 确保该目录已挂载到宿主机，或定期同步

## ? 最佳实践

### 推荐方案（最可靠）

```bash
# 1. 在宿主机上使用 docker exec -d 启动
docker exec -d <container_name> bash /root/workspace/mySAM/train.sh start

# 2. 设置容器 restart 策略
docker update --restart=unless-stopped <container_name>

# 3. 挂载数据目录到宿主机
docker run -v /host/data:/container/data ...

# 4. 定期检查训练状态
docker exec <container_name> bash /root/workspace/mySAM/train.sh status
```

### 快速检查脚本（在宿主机上运行）

```bash
#!/bin/bash
# check_training.sh - 在宿主机上运行

CONTAINER_NAME="your_container_name"

echo "检查训练状态..."
docker exec ${CONTAINER_NAME} bash /root/workspace/mySAM/train.sh status

echo ""
echo "查看最新日志..."
docker exec ${CONTAINER_NAME} bash /root/workspace/mySAM/train.sh logs | tail -20
```

## ? 常见问题

### Q: 退出容器后进程还在运行吗？
A: 如果使用 `bash train.sh start`（daemon 模式），进程会继续运行。但如果容器被停止，进程会终止。

### Q: 如何确保容器不会停止？
A: 
1. 使用 `docker exec -d` 启动进程
2. 设置 `--restart=unless-stopped` 策略
3. 使用 `Ctrl+P, Ctrl+Q` 退出容器（不停止）

### Q: 容器重启后如何恢复训练？
A: 
1. 使用 Docker restart 策略自动重启容器
2. 在容器启动脚本中自动启动训练
3. 或手动重新启动：`bash train.sh start`

### Q: 如何从宿主机查看训练日志？
A: 
```bash
docker exec <container_name> bash /root/workspace/mySAM/train.sh follow
```

### Q: 如何停止训练？
A: 
```bash
# 在容器内
bash train.sh stop

# 或从宿主机
docker exec <container_name> bash /root/workspace/mySAM/train.sh stop
```

### Q: 如何修改训练参数？
A: 
```bash
# 停止当前训练
bash train.sh stop

# 使用新参数启动
bash train.sh start --batch-size 1 --use-amp --use-gc
```

## ? 使用示例

### 示例1：最小显存配置

```bash
bash train.sh start \
  --batch-size 1 \
  --use-amp \
  --use-gc \
  --grad-accum 4
```

### 示例2：平衡配置（推荐）

```bash
bash train.sh start \
  --batch-size 2 \
  --use-amp \
  --grad-accum 2
```

### 示例3：使用 Screen 模式，方便查看

```bash
bash train.sh start --mode screen --batch-size 2 --use-amp
```

### 示例4：从宿主机启动

```bash
docker exec -d <container_name> bash -c "
  cd /root/workspace/mySAM && 
  bash train.sh start --batch-size 2 --use-amp
"
```
