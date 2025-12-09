#!/bin/bash
# 统一训练脚本 - 支持启动、停止、查看状态等功能
# 使用方法：bash train.sh [命令] [选项]

# 默认配置
DATA_ROOT="./data/ISIC"
TRAIN_BOX_CSV="./data/ISIC/train_boxes.csv"
SAM_CHECKPOINT="./checkpoints/sam_vit_h_4b8939.pth"
MODEL_TYPE="vit_h"
UNFREEZE_LAST_K=2
BATCH_SIZE=2
USE_AMP=""
USE_GRADIENT_CHECKPOINTING=""
GRADIENT_ACCUMULATION_STEPS=2
EPOCHS=30
OUTPUT_DIR="./outputs"
LOG_DIR="./logs"
MODE="daemon"  # daemon, screen, tmux, foreground

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 显示帮助信息
function show_help() {
    echo "统一训练脚本"
    echo ""
    echo "使用方法:"
    echo "  bash train.sh [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  start       - 启动训练（默认：守护进程模式）"
    echo "  stop        - 停止训练"
    echo "  status      - 查看训练状态"
    echo "  logs        - 查看最新训练日志"
    echo "  follow      - 实时跟踪训练日志"
    echo "  list        - 列出所有训练日志"
    echo "  clean       - 清理旧的日志文件"
    echo "  help        - 显示此帮助信息"
    echo ""
    echo "启动选项（start 命令）:"
    echo "  --mode MODE         运行模式: daemon(默认), screen, tmux, foreground"
    echo "  --batch-size N      批次大小（默认: 2）"
    echo "  --use-amp           启用混合精度训练"
    echo "  --use-gc            启用梯度检查点"
    echo "  --grad-accum N       梯度累积步数（默认: 2）"
    echo "  --unfreeze-k N       解冻最后K层（默认: 2）"
    echo "  --epochs N          训练轮数（默认: 30）"
    echo ""
    echo "示例:"
    echo "  bash train.sh start                    # 默认守护进程模式"
    echo "  bash train.sh start --mode screen      # 使用 screen 模式"
    echo "  bash train.sh start --batch-size 1 --use-amp  # 自定义参数"
    echo "  bash train.sh status                  # 查看状态"
    echo "  bash train.sh follow                  # 实时查看日志"
    echo "  bash train.sh stop                     # 停止训练"
}

# 解析启动参数
function parse_start_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --use-amp)
                USE_AMP="--use_amp"
                shift
                ;;
            --use-gc)
                USE_GRADIENT_CHECKPOINTING="--use_gradient_checkpointing"
                shift
                ;;
            --grad-accum)
                GRADIENT_ACCUMULATION_STEPS="$2"
                shift 2
                ;;
            --unfreeze-k)
                UNFREEZE_LAST_K="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# 启动训练
function start_training() {
    # 解析参数
    parse_start_args "$@"
    
    # 设置环境变量
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # 创建日志目录
    mkdir -p ${LOG_DIR}
    
    # 生成日志文件名
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
    PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"
    
    # 构建训练命令
    TRAIN_CMD="python model.py \
      --data_root ${DATA_ROOT} \
      --train_box_csv ${TRAIN_BOX_CSV} \
      --sam_checkpoint ${SAM_CHECKPOINT} \
      --model_type ${MODEL_TYPE} \
      --unfreeze_last_k ${UNFREEZE_LAST_K} \
      --batch_size ${BATCH_SIZE} \
      --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
      --epochs ${EPOCHS} \
      --output_dir ${OUTPUT_DIR} \
      ${USE_AMP} \
      ${USE_GRADIENT_CHECKPOINTING}"
    
    echo "=========================================="
    echo "启动训练"
    echo "模式: ${MODE}"
    echo "日志文件: ${LOG_FILE}"
    echo "PID 文件: ${PID_FILE}"
    echo "=========================================="
    
    case ${MODE} in
        daemon)
            # 守护进程模式：使用 setsid 创建新会话
            setsid bash -c "
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            cd $(pwd)
            ${TRAIN_CMD} > ${LOG_FILE} 2>&1
            " &
            TRAIN_PID=$!
            echo ${TRAIN_PID} > ${PID_FILE}
            echo -e "${GREEN}训练进程已启动，PID: ${TRAIN_PID}${NC}"
            echo "进程已使用 setsid 启动，退出 shell 不会终止训练"
            ;;
            
        screen)
            # Screen 模式
            if ! command -v screen &> /dev/null; then
                echo -e "${YELLOW}screen 未安装，正在安装...${NC}"
                apt-get update && apt-get install -y screen > /dev/null 2>&1
            fi
            SESSION_NAME="sam_train_${TIMESTAMP}"
            screen -dmS ${SESSION_NAME} bash -c "
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            ${TRAIN_CMD} > ${LOG_FILE} 2>&1
            "
            echo ${SESSION_NAME} > ${PID_FILE}
            echo -e "${GREEN}训练已在 screen 会话中启动: ${SESSION_NAME}${NC}"
            echo "重新连接: screen -r ${SESSION_NAME}"
            ;;
            
        tmux)
            # Tmux 模式
            if ! command -v tmux &> /dev/null; then
                echo -e "${YELLOW}tmux 未安装，正在安装...${NC}"
                apt-get update && apt-get install -y tmux > /dev/null 2>&1
            fi
            SESSION_NAME="sam_train_${TIMESTAMP}"
            tmux new-session -d -s ${SESSION_NAME} "
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            ${TRAIN_CMD} > ${LOG_FILE} 2>&1
            "
            echo ${SESSION_NAME} > ${PID_FILE}
            echo -e "${GREEN}训练已在 tmux 会话中启动: ${SESSION_NAME}${NC}"
            echo "重新连接: tmux attach -t ${SESSION_NAME}"
            ;;
            
        foreground)
            # 前台模式
            echo "在前台运行训练..."
            ${TRAIN_CMD}
            ;;
            
        *)
            echo -e "${RED}未知模式: ${MODE}${NC}"
            echo "支持的模式: daemon, screen, tmux, foreground"
            exit 1
            ;;
    esac
    
    echo ""
    echo "查看日志: tail -f ${LOG_FILE}"
    echo "查看状态: bash train.sh status"
}

# 停止训练
function stop_training() {
    PID_FILES=$(find ${LOG_DIR} -name "*.pid" 2>/dev/null | sort -r)
    
    if [ -z "$PID_FILES" ]; then
        echo -e "${YELLOW}未找到运行中的训练进程${NC}"
        return
    fi
    
    for PID_FILE in $PID_FILES; do
        PID=$(cat ${PID_FILE} 2>/dev/null)
        if [ -z "$PID" ]; then
            continue
        fi
        
        # 检查是否是 screen/tmux 会话名
        if [[ "$PID" == sam_train_* ]]; then
            # Screen 会话
            if screen -list | grep -q "$PID"; then
                echo "停止 screen 会话: ${PID}"
                screen -S ${PID} -X quit
            fi
            # Tmux 会话
            if tmux has-session -t ${PID} 2>/dev/null; then
                echo "停止 tmux 会话: ${PID}"
                tmux kill-session -t ${PID}
            fi
        else
            # 普通进程
            if ps -p ${PID} > /dev/null 2>&1; then
                echo "停止训练进程 (PID: ${PID})..."
                kill ${PID}
                sleep 2
                if ps -p ${PID} > /dev/null 2>&1; then
                    echo "强制停止..."
                    kill -9 ${PID}
                fi
                echo -e "${GREEN}训练已停止${NC}"
            fi
        fi
        rm -f ${PID_FILE}
    done
}

# 查看状态
function check_status() {
    echo "=========================================="
    echo "训练状态检查"
    echo "=========================================="
    
    PID_FILES=$(find ${LOG_DIR} -name "*.pid" 2>/dev/null | sort -r)
    
    if [ -z "$PID_FILES" ]; then
        echo -e "${YELLOW}未找到运行中的训练进程${NC}"
        return
    fi
    
    for PID_FILE in $PID_FILES; do
        PID=$(cat ${PID_FILE} 2>/dev/null)
        if [ -z "$PID" ]; then
            continue
        fi
        
        # 检查是否是 screen/tmux 会话
        if [[ "$PID" == sam_train_* ]]; then
            if screen -list | grep -q "$PID" || tmux has-session -t ${PID} 2>/dev/null; then
                echo -e "${GREEN}? 训练会话运行中${NC}"
                echo "  会话名: ${PID}"
                echo "  重新连接: screen -r ${PID} 或 tmux attach -t ${PID}"
            else
                echo -e "${RED}? 训练会话已停止${NC}"
            fi
        else
            if ps -p ${PID} > /dev/null 2>&1; then
                echo -e "${GREEN}? 训练进程运行中${NC}"
                echo "  PID: ${PID}"
                ps -p ${PID} -o pid,ppid,cmd,%mem,%cpu,etime
                
                # 显示 GPU 使用情况
                if command -v nvidia-smi &> /dev/null; then
                    echo ""
                    echo "GPU 使用情况:"
                    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -4
                fi
            else
                echo -e "${RED}? 训练进程已停止 (PID: ${PID})${NC}"
            fi
        fi
        
        # 显示日志文件
        LOG_FILE=$(echo ${PID_FILE} | sed 's/\.pid$/.log/')
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "日志文件: ${LOG_FILE}"
            echo "最新日志（最后3行）:"
            tail -3 ${LOG_FILE} | sed 's/^/  /'
        fi
        echo ""
    done
}

# 查看日志
function show_logs() {
    LATEST_LOG=$(find ${LOG_DIR} -name "*.log" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -z "$LATEST_LOG" ]; then
        echo -e "${YELLOW}未找到日志文件${NC}"
        return
    fi
    
    echo "=========================================="
    echo "最新训练日志: ${LATEST_LOG}"
    echo "=========================================="
    tail -50 ${LATEST_LOG}
}

# 实时跟踪日志
function follow_logs() {
    LATEST_LOG=$(find ${LOG_DIR} -name "*.log" -type f 2>/dev/null | sort -r | head -1)
    
    if [ -z "$LATEST_LOG" ]; then
        echo -e "${YELLOW}未找到日志文件${NC}"
        return
    fi
    
    echo "实时跟踪日志: ${LATEST_LOG}"
    echo "按 Ctrl+C 退出"
    echo ""
    tail -f ${LATEST_LOG}
}

# 列出所有日志
function list_logs() {
    echo "=========================================="
    echo "训练日志列表"
    echo "=========================================="
    
    LOG_FILES=$(find ${LOG_DIR} -name "*.log" -type f 2>/dev/null | sort -r)
    
    if [ -z "$LOG_FILES" ]; then
        echo "未找到日志文件"
        return
    fi
    
    for LOG_FILE in $LOG_FILES; do
        SIZE=$(du -h ${LOG_FILE} | cut -f1)
        MOD_TIME=$(stat -c %y ${LOG_FILE} 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
        echo ""
        echo "文件: ${LOG_FILE}"
        echo "  大小: ${SIZE}"
        echo "  修改时间: ${MOD_TIME}"
        LAST_LINE=$(tail -1 ${LOG_FILE} 2>/dev/null)
        if [ ! -z "$LAST_LINE" ]; then
            echo "  最新: ${LAST_LINE:0:100}..."
        fi
    done
}

# 清理旧日志
function clean_logs() {
    echo "=========================================="
    echo "清理旧日志"
    echo "=========================================="
    find ${LOG_DIR} -name "*.log" -type f -mtime +7 -delete
    find ${LOG_DIR} -name "*.pid" -type f -mtime +7 -delete
    echo "已清理7天前的日志文件"
}

# 主逻辑
case "$1" in
    start)
        shift
        start_training "$@"
        ;;
    stop)
        stop_training
        ;;
    status)
        check_status
        ;;
    logs)
        show_logs
        ;;
    follow)
        follow_logs
        ;;
    list)
        list_logs
        ;;
    clean)
        clean_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        if [ -z "$1" ]; then
            show_help
        else
            echo -e "${RED}未知命令: $1${NC}"
            echo ""
            show_help
            exit 1
        fi
        ;;
esac


