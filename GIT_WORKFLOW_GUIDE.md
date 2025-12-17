# Git 工作流程指南

本文档介绍如何将新项目与 Git 仓库同步，以及如何进行日常的更新维护。

## 目录

- [新项目初始化](#新项目初始化)
- [日常更新维护](#日常更新维护)
- [常用 Git 命令](#常用-git-命令)
- [最佳实践](#最佳实践)
- [常见问题处理](#常见问题处理)

---

## 新项目初始化

### 1. 创建本地 Git 仓库

如果项目还没有初始化为 Git 仓库：

```bash
# 进入项目目录
cd /path/to/your/project

# 初始化 Git 仓库
git init

# 查看状态
git status
```

### 2. 配置 Git 用户信息（首次使用需要）

```bash
# 设置全局用户名和邮箱（推荐）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 或者仅为当前项目设置
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 3. 创建 .gitignore 文件

在项目根目录创建 `.gitignore` 文件，排除不需要版本控制的文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.log

# 虚拟环境
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# 输出文件
outputs/
checkpoints/
*.pth
*.pt
*.ckpt

# 系统文件
.DS_Store
Thumbs.db
nohup.out

# 临时文件
*.tmp
*.bak
```

### 4. 添加文件到暂存区

```bash
# 添加所有文件（推荐先检查 git status）
git add .

# 或者选择性添加特定文件
git add file1.py file2.py

# 添加所有更改（包括删除的文件）
git add -A
```

### 5. 创建首次提交

```bash
# 提交更改
git commit -m "Initial commit: 项目初始化"

# 查看提交历史
git log
```

### 6. 连接到远程仓库

#### 方式一：已有远程仓库（推荐）

```bash
# 添加远程仓库地址
git remote add origin https://github.com/username/repository.git
# 或者使用 SSH
git remote add origin git@github.com:username/repository.git

# 查看远程仓库配置
git remote -v

# 推送到远程仓库（首次推送）
git push -u origin main
# 如果默认分支是 master，使用：
# git push -u origin master
```

#### 方式二：在 GitHub/GitLab 创建新仓库后

1. 在 GitHub/GitLab 上创建新仓库（不要初始化 README）
2. 复制仓库地址
3. 执行上述命令连接并推送

### 7. 验证同步状态

```bash
# 查看远程分支信息
git branch -r

# 查看本地和远程的同步状态
git status

# 拉取远程最新更改
git fetch origin
```

---

## 日常更新维护

### 标准工作流程

日常开发遵循以下流程：**修改 → 暂存 → 提交 → 推送**

#### 1. 查看当前状态

```bash
# 查看工作区状态
git status

# 查看详细的文件变更
git diff

# 查看已暂存的变更
git diff --staged
```

#### 2. 添加更改到暂存区

```bash
# 添加所有修改的文件
git add .

# 添加所有更改（包括删除）
git add -A

# 添加特定文件
git add path/to/file.py

# 添加特定目录
git add path/to/directory/

# 交互式添加（可以选择性添加部分更改）
git add -p
```

#### 3. 提交更改

```bash
# 提交暂存区的更改
git commit -m "描述性的提交信息"

# 提交信息规范示例：
# git commit -m "feat: 添加新的训练脚本"
# git commit -m "fix: 修复损失函数计算错误"
# git commit -m "docs: 更新 README 文档"
# git commit -m "refactor: 重构模型加载逻辑"
```

**提交信息规范（推荐）：**
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具相关

#### 4. 推送到远程仓库

```bash
# 推送到远程仓库
git push origin main

# 如果已经设置了上游分支，可以直接：
git push

# 首次推送时设置上游分支
git push -u origin main
```

#### 5. 拉取远程更新

在推送前，建议先拉取远程最新更改：

```bash
# 拉取并合并远程更改
git pull origin main

# 或者先获取，再合并（更安全）
git fetch origin
git merge origin/main

# 查看远程更改（不合并）
git fetch origin
git log origin/main
```

### 处理冲突

如果本地和远程都有更改，可能会产生冲突：

```bash
# 1. 先拉取远程更改
git pull origin main

# 2. 如果有冲突，Git 会提示哪些文件有冲突
# 打开冲突文件，会看到类似标记：
# <<<<<<< HEAD
# 你的本地更改
# =======
# 远程的更改
# >>>>>>> origin/main

# 3. 手动解决冲突后，标记为已解决
git add conflicted_file.py

# 4. 完成合并提交
git commit -m "Merge: 解决冲突"
```

---

## 常用 Git 命令

### 查看信息

```bash
# 查看工作区状态
git status

# 查看提交历史
git log
git log --oneline          # 简洁显示
git log --graph --oneline  # 图形化显示

# 查看文件变更
git diff                   # 工作区 vs 暂存区
git diff --staged          # 暂存区 vs 最后一次提交
git diff HEAD              # 工作区 vs 最后一次提交

# 查看远程仓库信息
git remote -v
git branch -a              # 查看所有分支（本地+远程）
```

### 撤销操作

```bash
# 撤销工作区的修改（未暂存）
git restore file.py
# 或使用旧命令
git checkout -- file.py

# 取消暂存（已 add 但未 commit）
git restore --staged file.py
# 或
git reset HEAD file.py

# 修改最后一次提交（未推送）
git commit --amend -m "新的提交信息"

# 撤销最后一次提交（保留文件更改）
git reset --soft HEAD~1

# 完全撤销最后一次提交（丢弃更改）
git reset --hard HEAD~1    # 危险操作，谨慎使用！
```

### 分支操作

```bash
# 查看分支
git branch                 # 本地分支
git branch -a              # 所有分支

# 创建分支
git branch new-feature

# 切换分支
git checkout new-feature
# 或使用新命令
git switch new-feature

# 创建并切换分支
git checkout -b new-feature
# 或
git switch -c new-feature

# 合并分支
git checkout main
git merge new-feature

# 删除分支
git branch -d new-feature  # 安全删除
git branch -D new-feature  # 强制删除
```

### 远程操作

```bash
# 添加远程仓库
git remote add origin <url>

# 查看远程仓库
git remote -v

# 重命名远程仓库
git remote rename origin upstream

# 删除远程仓库
git remote remove origin

# 拉取远程更改
git fetch origin
git pull origin main

# 推送更改
git push origin main

# 推送所有分支
git push --all origin

# 推送标签
git push --tags origin
```

---

## 最佳实践

### 1. 提交频率

- ✅ **频繁提交**：完成一个小功能或修复就提交一次
- ✅ **原子性提交**：每次提交只做一件事
- ❌ **避免**：积累大量更改后一次性提交

### 2. 提交信息

- ✅ **清晰描述**：提交信息要能清楚说明做了什么
- ✅ **使用规范**：遵循提交信息规范（feat/fix/docs 等）
- ❌ **避免**：使用 "update"、"fix bug" 等模糊信息

### 3. 工作前先拉取

```bash
# 开始工作前
git pull origin main

# 工作完成后
git add .
git commit -m "描述性信息"
git push origin main
```

### 4. 定期同步

- 每天开始工作前：`git pull`
- 完成功能后：立即 `git push`
- 避免本地积累过多未推送的提交

### 5. 使用分支

对于重要功能或实验性更改，使用分支：

```bash
# 创建功能分支
git checkout -b feature/new-model

# 在分支上开发
# ... 进行修改和提交 ...

# 完成后合并到主分支
git checkout main
git pull origin main
git merge feature/new-model
git push origin main

# 删除已合并的分支
git branch -d feature/new-model
```

### 6. .gitignore 管理

- 及时更新 `.gitignore`，排除不需要版本控制的文件
- 常见需要排除的：`__pycache__/`、`*.log`、`outputs/`、`checkpoints/` 等

---

## 常见问题处理

### 问题 1: 误提交了不应该提交的文件

```bash
# 从暂存区移除（但保留文件）
git restore --staged file.py

# 添加到 .gitignore
echo "file.py" >> .gitignore

# 重新提交
git add .gitignore
git commit -m "chore: 更新 .gitignore"
```

### 问题 2: 想撤销最后一次提交

```bash
# 保留文件更改，只撤销提交
git reset --soft HEAD~1

# 完全撤销（丢弃更改，危险！）
git reset --hard HEAD~1
```

### 问题 3: 已经推送了错误的提交

```bash
# 方法一：创建新提交来修复
git commit --amend -m "正确的提交信息"
git push --force origin main  # 谨慎使用！

# 方法二：创建撤销提交（更安全）
git revert HEAD
git push origin main
```

### 问题 4: 本地和远程都有更改

```bash
# 先拉取远程更改
git fetch origin

# 查看差异
git diff main origin/main

# 合并或变基
git pull origin main
# 或
git rebase origin/main
```

### 问题 5: 忘记添加文件到上次提交

```bash
# 添加遗漏的文件
git add forgotten_file.py

# 修改最后一次提交
git commit --amend --no-edit

# 如果已推送，需要强制推送（谨慎！）
git push --force origin main
```

### 问题 6: 查看某个文件的修改历史

```bash
# 查看文件的提交历史
git log --follow file.py

# 查看文件的具体变更
git log -p file.py

# 查看文件的每一行是谁修改的
git blame file.py
```

---

## 快速参考

### 新项目初始化流程

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <repository-url>
git push -u origin main
```

### 日常更新流程

```bash
git pull origin main          # 拉取最新更改
git status                    # 查看状态
git add .                     # 添加更改
git commit -m "描述信息"      # 提交
git push origin main          # 推送
```

### 紧急回退

```bash
git log --oneline             # 查看提交历史
git reset --hard <commit-id>  # 回退到指定提交（危险！）
git push --force origin main  # 强制推送（谨慎！）
```

---

## 总结

1. **新项目**：初始化 → 添加文件 → 提交 → 连接远程 → 推送
2. **日常维护**：拉取 → 修改 → 暂存 → 提交 → 推送
3. **遵循规范**：清晰的提交信息、频繁提交、使用分支
4. **安全第一**：推送前先拉取、谨慎使用 `--force`、重要操作前备份

---

**提示**：如果遇到不确定的操作，可以先在测试分支上尝试，或者查阅 Git 官方文档。

