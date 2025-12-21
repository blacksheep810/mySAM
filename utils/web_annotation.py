#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于Web的弱标注工具
通过浏览器访问，支持鼠标点击4个点生成外接大框
无需GUI支持，可在远程服务器上使用
"""

import os
import sys
import json
import base64
from typing import List, Tuple, Optional
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

# 全局配置
CONFIG = {
    'image_dir': None,
    'output_csv': None,
    'current_image_index': 0,
    'image_list': [],
    'annotations': {}  # {image_name: {'big_box': [x1,y1,x2,y2], 'points': [[x,y], ...]}}
}


def generate_small_box(big_box: List[int], image_path: str = None, method: str = 'contour', shrink_ratio: float = 0.7, min_size: int = 10) -> List[int]:
    """
    基于大框生成小框（智能方法）
    
    Args:
        big_box: 大框坐标 [x1, y1, x2, y2]
        image_path: 图像路径（如果提供，使用智能方法）
        method: 方法 ('contour', 'gradient', 'simple')
        shrink_ratio: 回退方法的收缩比例
        min_size: 最小尺寸
    
    Returns:
        [x1, y1, x2, y2]: 小框坐标
    """
    used_method = 'simple'  # 记录使用的方法
    
    # 如果提供了图像路径，使用智能方法
    if image_path and os.path.exists(image_path) and method != 'simple':
        try:
            # 尝试多种导入方式，兼容不同的运行环境
            try:
                # 方式1: 相对导入（作为包的一部分运行时）
                from .smart_box_generator import generate_smart_small_box
            except ImportError:
                try:
                    # 方式2: 绝对导入（从项目根目录运行时）
                    from utils.smart_box_generator import generate_smart_small_box
                except ImportError:
                    # 方式3: 直接导入（同目录下运行时）
                    import sys
                    import os
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    from smart_box_generator import generate_smart_small_box
            
            result = generate_smart_small_box(image_path, big_box, method=method)
            used_method = method  # 智能方法成功
            return result, used_method
        except Exception as e:
            print(f"警告: 智能方法失败，使用简单方法: {e}")
            used_method = f'simple (fallback from {method})'
    
    # 回退到简单方法（按比例缩小）
    x1, y1, x2, y2 = big_box
    
    # 确保坐标顺序正确
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # 计算中心点和尺寸
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # 计算小框尺寸
    small_width = max(width * shrink_ratio, min_size)
    small_height = max(height * shrink_ratio, min_size)
    
    # 计算小框坐标
    small_x1 = int(center_x - small_width / 2.0)
    small_y1 = int(center_y - small_height / 2.0)
    small_x2 = int(center_x + small_width / 2.0)
    small_y2 = int(center_y + small_height / 2.0)
    
    # 确保小框在大框内部
    small_x1 = max(small_x1, x1)
    small_y1 = max(small_y1, y1)
    small_x2 = min(small_x2, x2)
    small_y2 = min(small_y2, y2)
    
    return [max(0, small_x1), max(0, small_y1), small_x2, small_y2], used_method


def image_to_base64(image_path: str) -> str:
    """将图像转换为base64编码"""
    with open(image_path, 'rb') as f:
        img_data = f.read()
        base64_data = base64.b64encode(img_data).decode('utf-8')
        ext = os.path.splitext(image_path)[1][1:].lower()
        if ext == 'jpg':
            ext = 'jpeg'
        return f"data:image/{ext};base64,{base64_data}"


# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>弱标注工具 - Web版</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }
        .info {
            flex: 1;
            margin-right: 20px;
        }
        .info-item {
            margin: 5px 0;
            color: #666;
        }
        .buttons {
            display: flex;
            gap: 10px;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }
        .btn-primary {
            background: #007bff;
            color: white;
        }
        .btn-primary:hover {
            background: #0056b3;
        }
        .btn-success {
            background: #28a745;
            color: white;
        }
        .btn-success:hover {
            background: #218838;
        }
        .btn-warning {
            background: #ffc107;
            color: #333;
        }
        .btn-warning:hover {
            background: #e0a800;
        }
        .btn-danger {
            background: #dc3545;
            color: white;
        }
        .btn-danger:hover {
            background: #c82333;
        }
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        .btn-secondary:hover {
            background: #5a6268;
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
            background: #f5f5f5;
            border-radius: 5px;
            overflow: visible;
            position: relative;
            padding: 10px;
        }
        #imageCanvas {
            max-width: 100%;
            height: auto;
            cursor: crosshair;
            display: block;
            margin: 0 auto;
            border: 2px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            background: #fff;
        }
        #imageCanvas:hover {
            border-color: #007bff;
            box-shadow: 0 4px 12px rgba(0,123,255,0.2);
        }
        .instructions {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
        }
        .instructions h3 {
            color: #007bff;
            margin-bottom: 10px;
        }
        .instructions ol {
            margin-left: 20px;
        }
        .instructions li {
            margin: 5px 0;
            color: #333;
        }
        .status {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.danger {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .file-path {
            font-family: monospace;
            background: rgba(255, 255, 255, 0.3);
            padding: 2px 6px;
            border-radius: 3px;
            word-break: break-all;
            margin-top: 5px;
            display: block;
        }
        .points-list {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
        .point-item {
            display: inline-block;
            margin: 5px;
            padding: 5px 10px;
            background: white;
            border-radius: 3px;
            border: 1px solid #dee2e6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 弱标注工具 - Web版</h1>
        
        <div class="instructions">
            <h3>使用说明</h3>
            <ol>
                <li><strong>鼠标移动</strong>：移动鼠标时会显示<strong>黄色辅助线（横线和竖线）</strong>和坐标，帮助精确定位</li>
                <li>在图像上<strong>点击4个点</strong>（建议点击目标区域的边界附近，不一定要是四个角）</li>
                <li>点击的点会显示为<strong>绿色圆点</strong>，并标注序号（1-4）</li>
                <li>点击完成后，会自动计算并显示<strong>外接大框（红色）</strong>和<strong>内接小框（绿色）</strong></li>
                <li><strong>框的生成原理</strong>：系统会找到所有4个点的<strong>最小外接矩形</strong>（轴对齐矩形），所以即使点不在角上，也会生成一个包含所有点的矩形框</li>
                <li>点击<strong>"重置"</strong>按钮可以清除当前标注</li>
                <li>点击<strong>"保存并下一张"</strong>保存当前标注并跳转到下一张图像</li>
                <li>点击<strong>"保存CSV"</strong>将所有标注保存到CSV文件</li>
            </ol>
            <div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 3px;">
                <strong>💡 提示：</strong>点击的4个点不一定要是框的四个角。系统会自动计算包含所有点的最小矩形框。
                例如：如果点击的点是 (100,100), (200,150), (150,200), (180,180)，生成的框会是 [100, 100, 200, 200]。
            </div>
        </div>
        
        <div id="status"></div>
        
        <div class="controls">
            <div class="info">
                <div class="info-item"><strong>当前图像:</strong> <span id="currentImageName">-</span></div>
                <div class="info-item"><strong>进度:</strong> <span id="progress">0/0</span></div>
                <div class="info-item"><strong>已标注:</strong> <span id="annotatedCount">0</span> 张</div>
                <div class="points-list">
                    <strong>已点击的点:</strong> <span id="pointsList"></span>
                </div>
            </div>
            <div class="buttons">
                <button class="btn-warning" onclick="resetAnnotation()">重置</button>
                <button class="btn-primary" onclick="prevImage()">上一张</button>
                <button class="btn-primary" onclick="nextImage()">下一张</button>
                <button class="btn-success" onclick="saveAndNext()">保存并下一张</button>
                <button class="btn-danger" onclick="saveCSV()">保存CSV</button>
            </div>
        </div>
        
        <div class="image-container">
            <canvas id="imageCanvas"></canvas>
        </div>
    </div>

    <script>
        let currentImageIndex = 0;
        let imageList = [];
        let annotations = {};
        let points = [];
        let currentImageData = null;
        let canvas, ctx;

        // 事件监听器标志（防止重复绑定）
        let eventsBound = false;

        // 绑定事件监听器
        function bindEventListeners() {
            if (eventsBound || !canvas) {
                return;
            }
            
            console.log('绑定事件监听器...');
            
            // 鼠标移动事件 - 显示辅助线
            canvas.addEventListener('mousemove', function(e) {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                mouseX = Math.round((e.clientX - rect.left) * scaleX);
                mouseY = Math.round((e.clientY - rect.top) * scaleY);
                
                // 重新绘制（包括辅助线）
                redrawCanvas();
            });

            // 鼠标离开画布 - 清除辅助线
            canvas.addEventListener('mouseleave', function() {
                mouseX = -1;
                mouseY = -1;
                redrawCanvas();
            });

            // 鼠标点击事件
            canvas.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                
                console.log('Canvas点击事件触发', e);
                
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const x = Math.round((e.clientX - rect.left) * scaleX);
                const y = Math.round((e.clientY - rect.top) * scaleY);
                
                console.log(`点击坐标: (${x}, ${y}), 当前点数: ${points.length}`);
                
                // 限制最多4个点
                if (points.length >= 4) {
                    showStatus('已达到4个点，请先重置', 'info');
                    return;
                }
                
                points.push([x, y]);
                updatePointsList();
                
                // 重新绘制
                redrawCanvas();
                
                // 显示反馈
                showStatus(`已添加点 ${points.length}/4: (${x}, ${y})`, 'success');
                
                // 如果有4个点，计算并绘制框
                if (points.length === 4) {
                    console.log('已点击4个点，开始计算框...');
                    calculateAndDrawBoxes();
                }
            });
            
            eventsBound = true;
            console.log('事件监听器绑定完成');
        }

        // 初始化
        window.onload = function() {
            console.log('页面加载完成，开始初始化...');
            
            canvas = document.getElementById('imageCanvas');
            if (!canvas) {
                console.error('找不到canvas元素');
                showStatus('初始化失败: 找不到canvas元素', 'error');
                return;
            }
            
            ctx = canvas.getContext('2d');
            if (!ctx) {
                console.error('无法获取canvas上下文');
                showStatus('初始化失败: 无法获取canvas上下文', 'error');
                return;
            }
            
            // 设置初始画布尺寸（临时，加载图像后会更新）
            canvas.width = 800;
            canvas.height = 600;
            
            // 显示加载提示
            ctx.fillStyle = '#333';
            ctx.font = '20px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('正在加载图像列表...', canvas.width / 2, canvas.height / 2);
            ctx.textAlign = 'left';
            
            // 绑定事件监听器
            bindEventListeners();
            
            loadImageList();
        };

        // 加载图像列表
        function loadImageList() {
            fetch('/api/image_list')
                .then(response => response.json())
                .then(data => {
                    imageList = data.images;
                    currentImageIndex = data.current_index;
                    updateProgress();
                    loadCurrentImage();
                })
                .catch(error => {
                    showStatus('错误: ' + error, 'error');
                });
        }

        // 加载当前图像
        function loadCurrentImage() {
            if (imageList.length === 0) {
                showStatus('没有找到图像文件', 'error');
                return;
            }
            
            const imageName = imageList[currentImageIndex];
            console.log('加载图像:', imageName);
            showStatus('正在加载图像...', 'info');
            
            // 先清空当前点（防止残留）
            points = [];
            updatePointsList();
            
            fetch(`/api/image/${encodeURIComponent(imageName)}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    console.log('图像数据接收成功');
                    currentImageData = data;
                    displayImage(data.image_data);
                    // 不自动加载已有标注，每次都是全新的标注会话
                    // loadAnnotation(imageName);
                    updateProgress();
                    showStatus('图像加载成功', 'success');
                })
                .catch(error => {
                    console.error('加载图像失败:', error);
                    showStatus('加载图像失败: ' + error.message, 'error');
                    // 显示错误信息在画布上
                    if (canvas && ctx) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = '#333';
                        ctx.font = '20px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText('图像加载失败', canvas.width / 2, canvas.height / 2 - 20);
                        ctx.font = '14px Arial';
                        ctx.fillText(error.message, canvas.width / 2, canvas.height / 2 + 10);
                        ctx.textAlign = 'left';
                    }
                });
        }

        // 统一绘制函数（包括辅助线）
        function redrawCanvas() {
            if (!currentImageData || !currentImageData.image_data) {
                console.warn('没有图像数据，无法绘制');
                return;
            }
            
            const img = new Image();
            img.onerror = function() {
                console.error('图像加载失败');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '20px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('图像加载失败', canvas.width / 2 || 400, canvas.height / 2 || 300);
                ctx.textAlign = 'left';
            };
            
            img.onload = function() {
                // 设置画布尺寸为图像的实际尺寸
                canvas.width = img.width;
                canvas.height = img.height;
                
                console.log('设置画布尺寸:', canvas.width, 'x', canvas.height);
                
                // 清除画布
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // 绘制图像（完整绘制）
                ctx.drawImage(img, 0, 0, img.width, img.height);
                
                // 绘制辅助线（横线和竖线）
                if (mouseX >= 0 && mouseY >= 0) {
                    ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';  // 黄色半透明
                    ctx.lineWidth = 1;
                    ctx.setLineDash([5, 5]);  // 虚线
                    
                    // 横线
                    ctx.beginPath();
                    ctx.moveTo(0, mouseY);
                    ctx.lineTo(canvas.width, mouseY);
                    ctx.stroke();
                    
                    // 竖线
                    ctx.beginPath();
                    ctx.moveTo(mouseX, 0);
                    ctx.lineTo(mouseX, canvas.height);
                    ctx.stroke();
                    
                    ctx.setLineDash([]);  // 恢复实线
                    
                    // 显示坐标
                    ctx.fillStyle = 'yellow';
                    ctx.font = '12px Arial';
                    ctx.fillText(`(${mouseX}, ${mouseY})`, mouseX + 10, mouseY - 10);
                }
                
                // 绘制已有点
                points.forEach((p, i) => {
                    // 外圈（白色）
                    ctx.fillStyle = 'white';
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], 8, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // 内圈（绿色）
                    ctx.fillStyle = 'green';
                    ctx.beginPath();
                    ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // 编号
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText((i + 1).toString(), p[0], p[1]);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'alphabetic';
                });
                
                // 绘制已保存的框
                const imageName = imageList[currentImageIndex];
                if (annotations[imageName]) {
                    const ann = annotations[imageName];
                    if (ann.big_box) {
                        drawBox(ann.big_box, 'red', 3);
                    }
                    if (ann.small_box) {
                        drawBox(ann.small_box, 'green', 2);
                    }
                }
            };
            img.src = currentImageData.image_data;
        }

        // 显示图像
        function displayImage(imageData) {
            if (!imageData) {
                console.error('图像数据为空');
                showStatus('图像数据为空', 'error');
                return;
            }
            
            console.log('显示图像，数据长度:', imageData.length);
            currentImageData = {image_data: imageData};
            
            // 每次加载图像时清空点列表，不加载已有标注
            // 这样确保每次都是全新的标注会话
            points = [];
            updatePointsList();
            
            // 确保事件已绑定
            if (!eventsBound && canvas) {
                bindEventListeners();
            }
            
            // 绘制图像（不显示任何已保存的框）
            redrawCanvas();
        }

        // 绘制框
        function drawBox(box, color, lineWidth) {
            const [x1, y1, x2, y2] = box;
            ctx.strokeStyle = color;
            ctx.lineWidth = lineWidth;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }

        // 鼠标位置变量
        let mouseX = -1, mouseY = -1;

        // 计算并绘制框（只在用户点击4个点后调用）
        function calculateAndDrawBoxes() {
            if (points.length < 4) {
                console.warn('点数不足4个，无法计算框，当前点数:', points.length);
                return;
            }
            
            console.log('用户点击了4个点，开始计算框...');
            
            // 计算外接大框
            const xCoords = points.map(p => p[0]);
            const yCoords = points.map(p => p[1]);
            const bigBox = [
                Math.min(...xCoords),
                Math.min(...yCoords),
                Math.max(...xCoords),
                Math.max(...yCoords)
            ];
            
            // 调用后端API生成智能小框
            const imageName = imageList[currentImageIndex];
            fetch('/api/generate_small_box', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_name: imageName,
                    big_box: bigBox,
                    method: 'contour'  // 可选: 'contour', 'gradient', 'simple'
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }
                
                const smallBox = data.small_box;
                const methodUsed = data.method_used || 'unknown';
                
                // 保存标注
                annotations[imageName] = {
                    big_box: bigBox,
                    small_box: smallBox,
                    points: points.slice(),
                    method_used: methodUsed  // 保存使用的方法
                };
                
                // 重新绘制（包括框）
                redrawCanvas();
                
                // 显示详细信息和使用的办法
                const methodName = {
                    'contour': '轮廓检测方法',
                    'gradient': '梯度方法',
                    'max_inner': '最大内接矩形方法',
                    'simple': '简单方法（70%缩放）',
                    'simple (fallback from contour)': '简单方法（轮廓检测失败回退）',
                    'simple (fallback from gradient)': '简单方法（梯度方法失败回退）'
                }[methodUsed] || methodUsed;
                
                showStatus(`✅ 标注完成！大框: [${bigBox.join(', ')}] | 小框: [${smallBox.join(', ')}] | 使用方法: ${methodName}`, 'success');
            })
            .catch(error => {
                console.error('生成小框失败:', error);
                // 回退到简单方法
                const centerX = (bigBox[0] + bigBox[2]) / 2;
                const centerY = (bigBox[1] + bigBox[3]) / 2;
                const width = bigBox[2] - bigBox[0];
                const height = bigBox[3] - bigBox[1];
                const smallWidth = width * 0.7;
                const smallHeight = height * 0.7;
                
                const smallBox = [
                    Math.max(bigBox[0], centerX - smallWidth / 2),
                    Math.max(bigBox[1], centerY - smallHeight / 2),
                    Math.min(bigBox[2], centerX + smallWidth / 2),
                    Math.min(bigBox[3], centerY + smallHeight / 2)
                ];
                
                // 保存标注
                annotations[imageName] = {
                    big_box: bigBox,
                    small_box: smallBox,
                    points: points.slice()
                };
                
                // 重新绘制
                redrawCanvas();
                
                // 保存标注（包含方法信息）
                annotations[imageName] = {
                    big_box: bigBox,
                    small_box: smallBox,
                    points: points.slice(),
                    method_used: 'simple (fallback)'
                };
                
                showStatus('⚠️ 使用简单方法生成小框（智能方法失败）', 'info');
            });
        }

        // 更新点列表显示
        function updatePointsList() {
            const listEl = document.getElementById('pointsList');
            if (points.length === 0) {
                listEl.innerHTML = '<em>无</em>';
            } else {
                listEl.innerHTML = points.map((p, i) => 
                    `<span class="point-item">点${i+1}: (${p[0]}, ${p[1]})</span>`
                ).join('');
            }
        }

        // 更新进度
        function updateProgress() {
            document.getElementById('currentImageName').textContent = 
                imageList.length > 0 ? imageList[currentImageIndex] : '-';
            document.getElementById('progress').textContent = 
                `${currentImageIndex + 1}/${imageList.length}`;
            
            const annotatedCount = Object.keys(annotations).length;
            document.getElementById('annotatedCount').textContent = annotatedCount;
        }

        // 重置标注
        function resetAnnotation() {
            points = [];
            const imageName = imageList[currentImageIndex];
            delete annotations[imageName];
            updatePointsList();
            loadCurrentImage();
            showStatus('已重置', 'info');
        }

        // 上一张
        function prevImage() {
            if (currentImageIndex > 0) {
                currentImageIndex--;
                points = [];
                loadCurrentImage();
            }
        }

        // 下一张
        function nextImage() {
            if (currentImageIndex < imageList.length - 1) {
                currentImageIndex++;
                points = [];
                loadCurrentImage();
            }
        }

        // 保存并下一张
        function saveAndNext() {
            const imageName = imageList[currentImageIndex];
            if (!annotations[imageName] || !annotations[imageName].big_box) {
                showStatus('请先完成标注（点击4个点）', 'error');
                return;
            }
            
            fetch('/api/save_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_name: imageName,
                    annotation: annotations[imageName]
                })
            })
            .then(response => response.json())
            .then(data => {
                showStatus('保存成功！', 'success');
                setTimeout(() => {
                    if (currentImageIndex < imageList.length - 1) {
                        currentImageIndex++;
                        points = [];
                        loadCurrentImage();
                    } else {
                        showStatus('已经是最后一张图像', 'info');
                    }
                }, 500);
            })
            .catch(error => {
                showStatus('保存失败: ' + error, 'error');
            });
        }

        // 保存CSV
        function saveCSV() {
            fetch('/api/save_csv', {
                method: 'POST'
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.error || '保存失败');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    showStatus('保存CSV失败: ' + data.error, 'error');
                } else {
                    const fileInfo = `CSV文件已成功保存！(${data.count} 条记录${data.file_size ? ', 大小: ' + (data.file_size / 1024).toFixed(2) + ' KB' : ''})`;
                    showStatus(fileInfo, 'success', data.file_path);
                    console.log('CSV保存成功:', data);
                }
            })
            .catch(error => {
                console.error('保存CSV失败:', error);
                showStatus('保存CSV失败: ' + error.message, 'error');
            });
        }

        // 显示状态
        function showStatus(message, type, filePath = null) {
            const statusEl = document.getElementById('status');
            statusEl.innerHTML = message;
            if (filePath) {
                statusEl.innerHTML += '<span class="file-path">文件路径: ' + filePath + '</span>';
            }
            statusEl.className = 'status ' + (type === 'error' ? 'danger' : type);
            setTimeout(() => {
                statusEl.textContent = '';
                statusEl.className = 'status';
            }, 10000);  // 延长显示时间到10秒，方便查看文件路径
        }

        // 加载已保存的标注（只加载，不自动计算框）
        function loadAnnotation(imageName) {
            fetch(`/api/annotation/${encodeURIComponent(imageName)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.annotation) {
                        console.log('加载已保存的标注:', imageName, '有框:', !!data.annotation.big_box, '点数:', data.annotation.points ? data.annotation.points.length : 0);
                        // 只更新内存中的标注，不触发自动计算
                        annotations[imageName] = data.annotation;
                        // 如果有已保存的点，加载它们（但不自动计算框）
                        if (data.annotation.points && data.annotation.points.length > 0) {
                            points = data.annotation.points;
                            updatePointsList();
                            // 重新绘制以显示点和框（但不调用calculateAndDrawBoxes）
                            redrawCanvas();
                        } else if (data.annotation.big_box) {
                            // 如果有框但没有点，只显示框
                            redrawCanvas();
                        }
                    }
                })
                .catch(error => {
                    console.log('加载标注失败（可能没有保存的标注）:', error);
                    // 忽略错误，这是正常的
                });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """主页面"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/image_list')
def get_image_list():
    """获取图像列表"""
    return jsonify({
        'images': CONFIG['image_list'],
        'current_index': CONFIG['current_image_index']
    })


@app.route('/api/image/<image_name>')
def get_image(image_name):
    """获取图像数据"""
    image_path = os.path.join(CONFIG['image_dir'], image_name)
    if not os.path.exists(image_path):
        return jsonify({'error': '图像不存在'}), 404
    
    image_data = image_to_base64(image_path)
    return jsonify({
        'image_name': image_name,
        'image_data': image_data
    })


@app.route('/api/annotation/<image_name>')
def get_annotation(image_name):
    """获取标注"""
    annotation = CONFIG['annotations'].get(image_name)
    return jsonify({'annotation': annotation})


@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """保存标注"""
    data = request.json
    image_name = data['image_name']
    annotation = data['annotation']
    
    CONFIG['annotations'][image_name] = annotation
    return jsonify({'status': 'success'})


@app.route('/api/generate_small_box', methods=['POST'])
def generate_small_box_api():
    """API: 生成智能小框"""
    data = request.json
    image_name = data['image_name']
    big_box = data['big_box']
    method = data.get('method', 'contour')
    
    image_path = os.path.join(CONFIG['image_dir'], image_name)
    if not os.path.exists(image_path):
        return jsonify({'error': '图像不存在'}), 404
    
    try:
        result = generate_small_box(big_box, image_path=image_path, method=method)
        # 处理返回值（可能是元组或单个值）
        if isinstance(result, tuple):
            small_box, used_method = result
        else:
            small_box = result
            used_method = 'simple'
        
        return jsonify({
            'status': 'success',
            'small_box': small_box,
            'method_used': used_method  # 返回使用的方法
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_csv', methods=['POST'])
def save_csv():
    """保存CSV文件"""
    if not CONFIG['annotations']:
        return jsonify({'error': '没有标注数据'}), 400
    
    # 自动确定保存路径：如果图像目录在data/ISIC下，保存到data/ISIC目录
    image_dir = os.path.abspath(CONFIG['image_dir'])
    output_csv = CONFIG.get('output_csv', '')
    
    # 标准化路径分隔符
    image_dir_normalized = image_dir.replace('\\', '/')
    
    # 检查图像目录是否在data/ISIC下
    if 'data/ISIC' in image_dir_normalized or 'data\\ISIC' in image_dir_normalized:
        # 从路径中提取ISIC目录
        parts = image_dir_normalized.split('/')
        if 'ISIC' in parts:
            isic_idx = parts.index('ISIC')
            # 构建ISIC目录路径（使用绝对路径）
            isic_dir_parts = parts[:isic_idx+1]
            if isic_dir_parts[0] == '':
                # 绝对路径，以/开头
                isic_dir = '/' + '/'.join(isic_dir_parts[1:])
            else:
                # 相对路径
                isic_dir = os.path.join(*isic_dir_parts)
            isic_dir = os.path.abspath(isic_dir)
        else:
            # 回退：尝试从image_dir向上查找ISIC目录
            current_dir = image_dir
            isic_dir = None
            while current_dir != os.path.dirname(current_dir):
                if os.path.basename(current_dir) == 'ISIC':
                    isic_dir = current_dir
                    break
                current_dir = os.path.dirname(current_dir)
            
            if not isic_dir:
                # 如果找不到，使用默认路径
                isic_dir = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'ISIC')
                isic_dir = os.path.abspath(isic_dir)
        
        # 根据图像目录名称区分训练集和测试集
        dir_name = os.path.basename(image_dir)
        if 'Training' in image_dir or '训练' in image_dir or 'train' in dir_name.lower() or 'training' in dir_name.lower():
            dataset_type = 'train'
        elif 'Test' in image_dir or '测试' in image_dir or 'test' in dir_name.lower():
            dataset_type = 'test'
        else:
            dataset_type = 'annotations'
        
        # 生成带时间戳的文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_type == 'train':
            csv_filename = f"train_boxes_{timestamp}.csv"
        elif dataset_type == 'test':
            csv_filename = f"test_boxes_{timestamp}.csv"
        else:
            csv_filename = f"annotations_{timestamp}.csv"
        
        output_csv = os.path.join(isic_dir, csv_filename)
        # 确保目录存在
        os.makedirs(isic_dir, exist_ok=True)
        print(f"[INFO] 自动检测到ISIC目录: {isic_dir}, 数据集类型: {dataset_type}, 保存路径: {output_csv}")
    
    results = []
    for image_name, ann in CONFIG['annotations'].items():
        big_box = ann['big_box']
        # 使用智能方法生成小框
        image_path = os.path.join(CONFIG['image_dir'], image_name)
        if 'small_box' in ann:
            small_box = ann['small_box']
        else:
            result = generate_small_box(big_box, image_path=image_path, method='contour')
            small_box = result[0] if isinstance(result, tuple) else result
        
        results.append({
            'image_file': image_name,
            'mask_file': '',  # 占位符
            'max_boxes_x1': big_box[0],
            'max_boxes_y1': big_box[1],
            'max_boxes_x2': big_box[2],
            'max_boxes_y2': big_box[3],
            'min_boxes_x1': small_box[0],
            'min_boxes_y1': small_box[1],
            'min_boxes_x2': small_box[2],
            'min_boxes_y2': small_box[3],
        })
    
    df = pd.DataFrame(results)
    
    # 使用绝对路径保存
    output_csv = os.path.abspath(output_csv)
    
    try:
        df.to_csv(output_csv, index=False)
        
        # 验证文件是否真的保存成功
        if not os.path.exists(output_csv):
            raise FileNotFoundError(f"文件保存失败: {output_csv}")
        
        # 检查文件大小，确保不是空文件
        file_size = os.path.getsize(output_csv)
        if file_size == 0:
            raise ValueError(f"保存的文件为空: {output_csv}")
        
        print(f"[INFO] CSV文件已成功保存: {output_csv} (大小: {file_size} 字节, 记录数: {len(results)})")
        
        # 更新CONFIG中的output_csv路径
        CONFIG['output_csv'] = output_csv
        
        return jsonify({
            'status': 'success',
            'file_path': output_csv,
            'count': len(results),
            'file_size': file_size
        })
    except Exception as e:
        error_msg = f"保存CSV文件失败: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            'error': error_msg,
            'file_path': output_csv
        }), 500


def init_app(image_dir: str, output_csv: str, port: int = 5000, host: str = '0.0.0.0'):
    """
    初始化并启动Web应用
    
    Args:
        image_dir: 图像目录路径
        output_csv: 输出CSV文件路径
        port: 端口号（默认5000）
        host: 主机地址（默认0.0.0.0，允许外部访问）
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    
    # 获取图像列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_list = [
        f for f in os.listdir(image_dir)
        if any(f.lower().endswith(ext) for ext in image_extensions)
    ]
    image_list.sort()
    
    if not image_list:
        raise ValueError(f"图像目录中没有找到图像文件: {image_dir}")
    
    # 不自动加载已有标注，避免显示不需要的box
    # 用户需要手动标注，每次启动都是全新的标注会话
    
    CONFIG['image_dir'] = image_dir
    CONFIG['output_csv'] = output_csv
    CONFIG['image_list'] = image_list
    
    print("=" * 60)
    print("Web弱标注工具已启动")
    print("=" * 60)
    print(f"图像目录: {image_dir}")
    print(f"图像数量: {len(image_list)}")
    print(f"输出CSV: {output_csv}")
    print(f"已加载标注: {len(CONFIG['annotations'])} 张")
    print(f"\n请在浏览器中访问: http://{host}:{port}")
    print(f"或访问: http://localhost:{port}")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web弱标注工具")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    parser.add_argument("--output_csv", type=str, required=True, help="输出CSV文件路径")
    parser.add_argument("--port", type=int, default=5000, help="端口号（默认5000）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址（默认0.0.0.0）")
    
    args = parser.parse_args()
    
    init_app(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        port=args.port,
        host=args.host
    )

