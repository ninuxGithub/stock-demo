# stock-demo

这是一个示例仓库，包含 `stock_selector.py` 模块，用于获取A股实时行情、热门股票信息，并基于技术评分与热度进行选股。

## 依赖

- Python 3.9+
- pandas
- akshare

可使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 运行方式

直接运行脚本：

```bash
python stock_selector.py
```

可选参数：

- `--top-n`：最终输出股票数量，默认 `30`
- `--universe-n`：从前多少名成交额股票中选股，默认 `200`
- `--no-fa-filter`：禁用成交额筛选

示例：

```bash
python stock_selector.py --top-n 20 --universe-n 100
```
