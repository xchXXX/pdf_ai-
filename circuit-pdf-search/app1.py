from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pdfplumber
import os
from pathlib import Path
import time
from PIL import Image
import pytesseract
import re
from PIL import  ImageFilter

# -------------------------- 核心配置：支持pdfs文件夹下所有PDF --------------------------
app = FastAPI(title="多PDF文档解析服务", description="自动识别pdfs文件夹下所有PDF文件")

# 1. 静态文件配置（无改动）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. PDF目录配置（固定为pdfs文件夹，自动识别所有PDF）
PDF_DIR = "pdfs"
Path(PDF_DIR).mkdir(exist_ok=True)  # 确保目录存在


# 3. 动态获取PDF列表（核心改动：自动扫描pdfs文件夹）
def get_all_pdfs():
    """获取pdfs文件夹下所有PDF文件"""
    pdfs = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdfs.append({
                "name": filename,
                "path": os.path.join(PDF_DIR, filename)
            })
    return pdfs


# 4. 解析状态与缓存（改为以文件名作为键，支持多文件）
parse_status = {}  # 结构: {filename: {status, progress, error}}
pdf_cache = {}  # 结构: {filename: {解析结果}}

# 初始化状态（启动时扫描一次）
for pdf in get_all_pdfs():
    if pdf["name"] not in parse_status:
        parse_status[pdf["name"]] = {
            "status": "pending",
            "progress": 0,
            "error": ""
        }


# -------------------------- 工具函数：保持原有逻辑，仅适配多文件 --------------------------
def is_title(text_chunk, page_chars, page_height):
    """判断文本块是否为标题（保持原有逻辑）"""
    if not text_chunk or len(text_chunk) < 3 or len(text_chunk) > 80:
        return False

    chunk_font_sizes = []
    for char in page_chars:
        if char["text"].strip() and char["text"] in text_chunk:
            chunk_font_sizes.append(char["size"])
    if not chunk_font_sizes:
        return False
    avg_chunk_size = sum(chunk_font_sizes) / len(chunk_font_sizes)

    all_font_sizes = [char["size"] for char in page_chars if char["text"].strip()]
    if not all_font_sizes:
        return False
    avg_page_size = sum(all_font_sizes) / len(all_font_sizes)

    top_positions = [char["top"] for char in page_chars if char["text"] in text_chunk]
    if not top_positions:
        return False
    min_top = min(top_positions)

    return avg_chunk_size > avg_page_size * 1.4 and min_top < page_height * 0.3


def optimize_tech_symbols(text):
    """技术符号优化（复用之前的符号映射逻辑）"""
    symbol_map = {
        # 通用汽车文档符号
        r'([AB]_GD\d+[A-Z]?)': r'接地\1 \1',
        r'(R\d+)': r'\1继电器 继电器\1',
        r'([B|P|T]-CAN-[H|L])': r'\1 CAN总线 CAN总线\1',
        r'(\d{2})\s*(\d{1,2})': r'\1-\2',
        # J6L专属符号
        r'(FA10-\w+)': r'\1 锡柴FA10气驱罐 气驱罐\1',
        r'气驱罐(\w+)': r'气驱罐\1 气驱罐组件 \1气驱罐',
        r'国六(\w+)': r'国六\1 国六后处理\1 \1国六'
    }
    for pattern, replacement in symbol_map.items():
        text = re.sub(pattern, replacement, text)
    return text


def parse_pdf_async(pdf_name):
    """异步解析指定PDF（从单文件改为接受pdf_name参数）"""
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        parse_status[pdf_name] = {
            "status": "failed",
            "progress": 0,
            "error": f"文件不存在：{pdf_path}"
        }
        return

    try:
        # 初始化状态
        parse_status[pdf_name] = {"status": "processing", "progress": 0, "error": ""}

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages = []

            for page_idx, page in enumerate(pdf.pages):
                # 更新进度
                current_progress = int((page_idx + 1) / total_pages * 100)
                parse_status[pdf_name]["progress"] = current_progress
                time.sleep(0.1)

                page_height = page.height
                page_width = page.width

                # 矢量文字提取
                vector_text = page.extract_text() or ""
                chars = page.chars or []
                words = page.extract_words() or []

                # 判断矢量文字是否有效
                # is_vector_valid = (
                #         len(vector_text.strip()) > 60
                #         or any(keyword in vector_text for keyword in ["线束", "继电器", "端子", "插接件", "ECU"])
                # )

                # OCR补充
                ocr_text = ""
                ocr_words = []
                #if not is_vector_valid:
                img = page.to_image(resolution=300)
                pil_img = img.original
                # 1. 彩色图转黑白二值图（工程图黑白对比强，OCR更易识别）
                pil_img = pil_img.convert('L')  # 转灰度图
                # 2. 二值化（替换ImageOps.threshold，用Image.point()实现）
                # 原理：像素值>127设为255（白色），≤127设为0（黑色），去除灰度干扰
                threshold = 127  # 阈值（可根据图像清晰度调整，127是中间值，适合大多数扫描图）
                pil_img = pil_img.point(lambda p: 255 if p > threshold else 0)

                # 2. 去噪（去除扫描时的小污渍、杂点）
                pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))  # 中值滤波去噪

                # 3. 放大图像（针对小字：如果线束图字特别小，放大1.5倍后识别）
                pil_img = pil_img.resize((int(pil_img.width * 2), int(pil_img.height * 2)),Image.Resampling.LANCZOS)
                custom_config = r'--oem 3 --psm 6 -l chi_sim+eng'
                ocr_text = pytesseract.image_to_string(pil_img, config=custom_config) or ""
                # 获取OCR文字坐标
                ocr_data = pytesseract.image_to_data(pil_img, config=custom_config,output_type=pytesseract.Output.DICT)
                for i in range(len(ocr_data["text"])):
                    text = ocr_data["text"][i].strip()
                    if text:
                        ocr_words.append({
                            "text": text,
                            "top": ocr_data["top"][i] * (page.height / pil_img.height),
                            "size": ocr_data["height"][i] * (page.height / pil_img.height),
                            "x0": ocr_data["left"][i] * (page.width / pil_img.width),
                            "x1": (ocr_data["left"][i] + ocr_data["width"][i]) * (page.width / pil_img.width)
                        })

                # 文本优化（合并矢量+OCR+符号优化）
                raw_text = f"{vector_text} {ocr_text}".strip()
                optimized_text = optimize_tech_symbols(raw_text)
                vertical_optimized_text = optimized_text.replace('\n', '').replace(' ', '')
                final_text = f"{raw_text} {optimized_text} {vertical_optimized_text}".strip()

                # 表格提取
                tables = page.extract_tables() or []
                table_texts = set()
                if tables:
                    for table in tables:
                        for row in table:
                            for cell in row:
                                if cell and isinstance(cell, str):
                                    for word in cell.replace('\n', ' ').split():
                                        if word.strip():
                                            table_texts.add(word.lower())

                # 标题识别
                title_texts = set()
                text_blocks = final_text.split('\n')
                for block in text_blocks:
                    block = block.strip()
                    if is_title(block, chars, page_height):
                        for word in block.split():
                            title_texts.add(word.lower())

                # 组织页面数据
                pages.append({
                    "page_num": page_idx + 1,
                    "text": final_text,
                    "words": words,
                    "title_texts": title_texts,
                    "table_texts": table_texts,
                    "page_info": f"第{page_idx + 1}页（宽：{page_width:.0f}px，高：{page_height:.0f}px）"
                })

            # 缓存结果
            pdf_cache[pdf_name] = {
                "total_pages": total_pages,
                "pages": pages,
                "pdf_name": pdf_name,
                "parse_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            parse_status[pdf_name] = {
                "status": "completed",
                "progress": 100,
                "error": ""
            }

    except Exception as e:
        error_msg = str(e)[:200]
        parse_status[pdf_name] = {
            "status": "failed",
            "progress": 0,
            "error": f"解析失败：{error_msg}"
        }
        print(f"PDF {pdf_name} 解析错误：{str(e)}")


# -------------------------- 接口：仅修改为支持多文件，逻辑不变 --------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")


@app.get("/pdfs")
async def get_pdf_list():
    """获取pdfs文件夹下所有PDF列表（动态生成）"""
    pdfs = get_all_pdfs()
    return {
        "pdfs": [
            {
                "name": pdf["name"],
                "size": os.path.getsize(pdf["path"]) if os.path.exists(pdf["path"]) else 0,
                "current_status": parse_status.get(pdf["name"], {"status": "pending"})["status"],
                "path": pdf["path"]
            } for pdf in pdfs
        ],
        "count": len(pdfs),
        "note": f"共发现{len(pdfs)}个PDF文件（位于pdfs文件夹）"
    }


@app.get("/viewer/{pdf_name}")
async def get_viewer(pdf_name: str):
    """查看器页面（增加pdf_name参数）"""
    return FileResponse("static/viewer.html")


@app.get("/pdf/{pdf_name}")
async def get_pdf_file(pdf_name: str):
    """获取指定PDF文件（增加pdf_name参数）"""
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    if not os.path.exists(pdf_path) or not pdf_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=404, detail=f"PDF文件不存在：{pdf_name}")
    return FileResponse(
        path=pdf_path,
        filename=pdf_name,
        media_type="application/pdf"
    )


@app.get("/search/{pdf_name}")
async def search_in_pdf(pdf_name: str, query: str):
    """搜索指定PDF（增加pdf_name参数，搜索逻辑不变）"""
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")

    status = parse_status[pdf_name]
    if status["status"] == "pending":
        raise HTTPException(status_code=400, detail=f"PDF尚未解析，请先触发解析：{pdf_name}")
    if status["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"解析失败：{status['error']}")

    pdf_data = pdf_cache.get(pdf_name)
    if not pdf_data:
        raise HTTPException(status_code=500, detail="解析结果缓存丢失")

    # 搜索逻辑与原代码一致
    results = []
    query_lower = query.strip().lower()
    if not query_lower:
        raise HTTPException(status_code=400, detail="搜索关键词不能为空")

    for page in pdf_data["pages"]:
        page_num = page["page_num"]
        text_lower = page["text"].lower()
        if query_lower not in text_lower:
            continue

        score = 1.0
        is_title_hit = any(query_lower in title_word for title_word in page["title_texts"])
        is_table_hit = any(query_lower in table_word for table_word in page["table_texts"])

        if is_title_hit:
            score += 4.0
        elif is_table_hit:
            score += 2.5

        # 位置提取
        position = None
        for word in page["words"]:
            word_text_lower = word["text"].lower()
            if query_lower in word_text_lower or word_text_lower in query_lower:
                position = {
                    "x": word["x0"], "y": word["top"],
                    "width": word["x1"] - word["x0"],
                    "height": word["bottom"] - word["top"],
                    "page_num": page_num
                }
                break

        # 上下文片段
        snippet_start = max(0, text_lower.find(query_lower) - 80)
        snippet_end = min(len(text_lower), snippet_start + 200)
        snippet = page["text"][snippet_start:snippet_end].replace('\n', ' ')
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(text_lower):
            snippet += "..."

        results.append({
            "page_num": page_num,
            "score": round(score, 1),
            "position": position,
            "snippet": snippet,
            "is_title": is_title_hit,
            "is_table": is_table_hit,
            "page_info": page["page_info"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "target_pdf": pdf_name,
        "query": query,
        "results": results,
        "total": len(results),
        "total_pages": pdf_data["total_pages"]
    }


@app.post("/parse-trigger/{pdf_name}")
async def trigger_parse(pdf_name: str, background_tasks: BackgroundTasks):
    """触发指定PDF解析（增加pdf_name参数）"""
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")

    current_status = parse_status[pdf_name]["status"]
    if current_status == "processing":
        return {
            "success": True,
            "message": f"正在解析：{pdf_name}",
            "status": current_status,
            "progress": parse_status[pdf_name]["progress"]
        }
    if current_status == "completed":
        return {
            "success": True,
            "message": f"已解析完成：{pdf_name}",
            "status": current_status,
            "parse_time": pdf_cache[pdf_name]["parse_time"]
        }

    background_tasks.add_task(parse_pdf_async, pdf_name)
    return {
        "success": True,
        "message": f"解析任务已启动：{pdf_name}",
        "status": "processing",
        "progress": 0
    }


@app.get("/parse-status/{pdf_name}")
async def get_parse_status(pdf_name: str):
    """查询指定PDF解析状态（增加pdf_name参数）"""
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    return {
        "target_pdf": pdf_name,
        "status": parse_status[pdf_name]["status"],
        "progress": parse_status[pdf_name]["progress"],
        "error": parse_status[pdf_name]["error"],
        "last_check_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# -------------------------- 服务启动（无改动） --------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        timeout_keep_alive=600
    )
