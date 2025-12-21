#!/bin/bash
# Web弱标注工具启动脚本

IMAGE_DIR="./data/ISIC/ISBI2016_ISIC_Part1_Training_Data"
# 自动生成带时间戳的CSV文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_CSV="./data/ISIC/annotations_${TIMESTAMP}.csv"
PORT=5000

echo "启动Web弱标注工具..."
echo "图像目录: $IMAGE_DIR"
echo "输出CSV: $OUTPUT_CSV"
echo "端口: $PORT"
echo ""
echo "访问地址:"
echo "  本地: http://localhost:$PORT"
echo "  远程: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "使用SSH端口转发（推荐）:"
echo "  ssh -L $PORT:localhost:$PORT user@server_ip"
echo "  然后在本地浏览器访问: http://localhost:$PORT"
echo ""

python utils/web_annotation.py \
    --image_dir "$IMAGE_DIR" \
    --output_csv "$OUTPUT_CSV" \
    --port "$PORT"
