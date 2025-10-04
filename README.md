# 暴力引文网络生成

基于scopus附带参考文献的csv导出进行暴力引文网络生成，生成html文件和csv表格，匹配精度受限于数据质量。

## 🛠️ 安装要求

- Python 3.11 或更高版本
- [uv](https://github.com/astral-sh/uv) 包管理工具

> ## On Windows.
>
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

## 使用 uv 创建虚拟环境并安装依赖

```bash
uv venv .venv
uv sync
```

## 基本用法

1. 参考env.example，设置.env为母文件夹位置，其中包含data文件夹存放数据，运行以下命令将自动生成output文件夹及其中的引文网络

2. 设置超参数，控制出度入度、判断逻辑和绘制孤立点

```bash
uv run citation_net_generation.py
```
