from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import time
import requests
import re
from pdf2image import convert_from_path  # PDF转图片（供AI识别）
from dotenv import load_dotenv  # 加载AI密钥（避免硬编码）
import pdfplumber
from PIL import Image
import langchain
from langchain_openai import ChatOpenAI,OpenAI
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
import io, base64

# 加载环境变量（AI密钥放在.env文件中）
load_dotenv()
app = FastAPI(title="AI-PDF电路图搜索系统", description="全AI实现文字提取与搜索定位")

# -------------------------- 基础配置（保留原有逻辑） --------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
PDF_DIR = "pdfs"
Path(PDF_DIR).mkdir(exist_ok=True)

# AI API配置（替换为你的密钥）
BAIDU_OCR_API_KEY = os.getenv("BAIDU_OCR_API_KEY", "你的百度智能云OCR API Key")
BAIDU_OCR_SECRET_KEY = os.getenv("BAIDU_OCR_SECRET_KEY", "你的百度智能云OCR Secret Key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "你的GPT API Key")
OPENAI_BASE_URL=os.getenv("OPENAI_BASE_URL", "你的GPT BASE URL")


# -------------------------- 原有状态管理（无改动） --------------------------
def get_all_pdfs():
    pdfs = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdfs.append({"name": filename, "path": os.path.join(PDF_DIR, filename)})
    return pdfs


parse_status = {}
pdf_cache = {}
for pdf in get_all_pdfs():
    if pdf["name"] not in parse_status:
        parse_status[pdf["name"]] = {"status": "pending", "progress": 0, "error": ""}


# -------------------------- 核心AI工具函数（新增） --------------------------
def get_baidu_ocr_token():
    """获取百度OCR接口的访问Token"""
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={BAIDU_OCR_API_KEY}&client_secret={BAIDU_OCR_SECRET_KEY}"
    response = requests.get(url)
    return response.json().get("access_token")


def ai_extract_text_with_position(img_path, token):
    """调用百度AI OCR（通用文字识别-高精度含位置版）"""
    import base64
    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()


    # 通用文字识别（高精度含位置版）路径
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate?access_token={token}"

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    # data = {
    #     "image": img_base64,
    #     "language_type": "CHN_ENG",  # 中英双语
    #     "detect_direction": "true",  # 检测方向
    #     "with_vertices": "true"  # 返回位置信息
    # }
    data = {
               "image": img_base64,
               "language_type": "CHN_ENG",  # 适配电路图中中文标注+英文缩写（如ECU、ABS）
               "detect_direction": "true",  # 矫正图像整体旋转（如电路图扫描倾斜），不影响文字行方向
               "with_vertices": "true",  # 输出文字坐标，便于核对竖直文字位置
               "detect_language": "true",  # 自动识别中英混合文字
               "paragraph": "true",  # 按段落聚合，避免边缘小文字漏检（如线号集群标注）
               "probability": "true",  # 返回置信度，排查低置信度的竖直小文字（如模糊线号）
               "min_size": 10,  # 匹配电路图小文字（线号通常8-12像素），避免漏检
               "text_direction": "auto",# 优先选择auto，无则删除此参数
               "recognize_granularity": "small", #开启细粒度识别，适配线号、引脚号等单个字符 / 短文本
               "filter_noise": "true" #过滤电路图中的线条干扰（如导线、符号），避免误将线条识别为文字
    }

    response = requests.post(url, data=data, headers=headers)
    result = response.json()
    print(result)

    if "error_code" in result:
        print(f"百度OCR错误详情：{result}")  # 打印完整错误信息
        raise Exception(f"高精度OCR识别失败：{result['error_msg']}（错误码：{result['error_code']}）")

    # -------------------------- 关键：适配高精度接口的位置字段 --------------------------
    texts = []
    positions = []
    for line in result.get("words_result", []):
        text = line["words"].strip()
        if not text:
            continue

        loc = line["location"]
        positions.append({
            "x0": loc["left"],  # 左上角X坐标
            "y0": loc["top"],  # 左上角Y坐标
            "width": loc["width"],  # 宽度（直接返回，无需计算）
            "height": loc["height"],  # 高度（直接返回，无需计算）
            "confidence": line.get("probability", {}).get("average", 0)  # 置信度
        })
        texts.append(text)

    return {
        "full_text": "\n".join(texts),
        "text_list": texts,
        "positions": positions,
        # 从位置信息中计算图片宽高（适配电路图场景）
        "img_width": max([p["x0"] + p["width"] for p in positions], default=0),
        "img_height": max([p["y0"] + p["height"] for p in positions], default=0)
    }


def ai_document_qa(pdf_name, question):
    """调用通义千问API，基于PDF内容回答连接性问题"""
    if pdf_name not in pdf_cache:
        return "PDF尚未解析，请先触发解析"
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    # 提取PDF所有页面的AI识别文本
    pdf_page_text = [page['ai_text'] for page in pdf_cache[pdf_name]["pages"]]
    chat_model = ChatOpenAI(
        model_name='gpt-4o-mini',
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )
    prompt=PromptTemplate.from_template("你是汽车电路图专家，从问题'{input}'中提取2-3个核心实体（仅保留术语，如连接器编号、针脚号，不要加“连接器”“针脚”等描述词），用英文逗号分割，结果无空格、无多余内容（例：问题“X123的A5针脚功能”→输出“X123,A5”）。")
    parser=StrOutputParser()
    response=chat_model.invoke(prompt.invoke(question))
    res=parser.invoke(response)
    items = [
        e.strip()  # 去除前后空格（解决“ X18 针脚”→“X18针脚”）
        for e in res.split(',')
        if e.strip() != ''  # 过滤空字符串
           and e.strip().lower() not in ["功能", "作用", "对应"]  # 排除无意义词
    ]
    related_pages_num = []
    for page_idx, page_text in enumerate(pdf_page_text):
        page_text_lower = page_text.lower()  # 转小写，忽略大小写差异
        if any(
                j.lower() in page_text_lower  # 实体小写后，匹配页面小写文本
                for j in items
                if j.strip() != ''  # 排除空实体
        ):
            related_pages_num.append(page_idx + 1)
    # 去重（避免同一页面被多次添加）
    related_pages_num = list(dict.fromkeys(related_pages_num))
    if not related_pages_num:
        return f"未找到包含{items}的页面"
    else:
        print(f'founed page:{related_pages_num}')
    image_base64_list=[]
    with pdfplumber.open(pdf_path) as pdf:
        for i in related_pages_num:
            # pdfplumber页码从0开始，需-1
            page = pdf.pages[i - 1]
            # 转为高清图片（300DPI）
            img = page.to_image(resolution=300)
            pil_img = img.original  # PIL Image对象
            # 图片转Base64（无需保存临时文件，直接内存处理）
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_base64_list.append({
                "page_num": i,
                "base64": img_base64
            })
    # 构造AI Prompt（强调基于电路图内容回答）
    url_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img['base64']}"}
            }
            for img in image_base64_list
        ]
    human_qes=[{"type":"text","text":question}]
    human_message=human_qes+url_content
    prompt = ChatPromptTemplate.from_messages(
        [("system", """
            你是汽车电路图专家，仅基于以下PDF电路图内容回答问题，不编造信息：
            1. PDF名称：{pdf_name}
            回答要求：
            - 若图片中包含清晰文字，优先基于文字信息回答；若包含图形（如针脚符号、导线），结合图形补充连接关系细节。
            - 明确给出答案（如针脚号、连接部件）；
            - 标注信息来源的页码；
            - 语言简洁，避免无关内容。
            """),
         ("human",human_message)
         ]
    )
    try:
        response=chat_model.invoke(prompt.invoke({"pdf_name": pdf_name}))
        return StrOutputParser().invoke(response)
    except Exception as e:
        return f"AI问答失败：{str(e)[:100]}"


# -------------------------- 同义词扩展（适配电路图元器件） --------------------------
def expand_synonyms(query):
    """搜索扩展：基于同义词映射表扩展查询词"""
    chat_model=ChatOpenAI(
        model_name = 'gpt-4o-mini',
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.1
    )
    query_lower = query.strip().lower()
    examples=[{'input':"油门踏板","output":"油门踏板,踏板位置传感器,Accelerator Pedal Sensor,APS"},
              {'input':"挂车控制模块","output":"挂车控制模块,挂车控制器,Trailer Control Module,TCM"},
              {'input':"ECU","output":"ECU,发动机控制器,Engine Control Unit"},
              {'input':"制动蹄片","output":"制动蹄片,刹车片,Brake Shoe"},
              {'input':"尿素液位","output":"尿素液位,尿素量,Urea Level Sensor"},
              {'input':"CAN总线","output":"CAN总线,CAN-H,CAN-L,CAN总线信号"},]
    ex_prompt=ChatPromptTemplate.from_messages(
        [("human","{input}的同义词有哪些"),
        ("ai","{output}")]
    )
    few_shot_prompt=FewShotChatMessagePromptTemplate(
        example_prompt=ex_prompt,
        examples=examples
    )
    final_shot_prompt=ChatPromptTemplate.from_messages(
        [("system","你是一名资深30年的汽车元器件领域专家,深入了解各种元器件以及其所含有相同意义的同义词,同时对于普通的词汇也能知道其相应的同义词,输出要求：仅返回同义词列表，用英文逗号分割，无任何前缀、无多余解释,参考示例，若无可扩展同义词则返回原词本身."),
         few_shot_prompt,
         ("human","{input}")])
    parser = StrOutputParser()
    response=chat_model.invoke(final_shot_prompt.invoke(query_lower))
    res=parser.invoke(response).strip()
    expanded_words=[i.strip() for i in res.split(',') if i.strip()]
    print(expanded_words)
    return expanded_words

# 新增：压缩图片尺寸和大小
def compress_image(pil_img,temp_img_path, max_size=3.8):
    """
    压缩图片至指定大小以下（单位：MB），同时限制最大分辨率
    """
    # 1. 限制最大分辨率（防止超宽/超高图片）
    max_dimension = 3000  # 最大宽或高（像素）
    width, height = pil_img.size
    new_size=(width, height)
    if width > max_dimension or height > max_dimension:
        # 按比例缩放
        ratio = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * ratio), int(height * ratio))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # 2. 压缩文件大小（通过调整质量）
    quality = 95  # 初始质量
    while True:
        # 临时保存到内存，检查大小
        import io
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='PNG', quality=quality)
        img_size = img_byte_arr.tell() / 1024 / 1024  # 转换为MB

        if img_size <= max_size or quality <= 50:
            # 达到目标大小或质量过低时停止（最低保留50%质量）
            pil_img.save(temp_img_path, format='PNG', quality=quality)
            break
        quality -= 5  # 每次降低5%质量
    return new_size

# -------------------------- AI解析PDF--------------------------
def parse_pdf_async(pdf_name):
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        parse_status[pdf_name] = {"status": "failed", "progress": 0, "error": f"文件不存在：{pdf_path}"}
        return

    try:
        parse_status[pdf_name] = {"status": "processing", "progress": 0, "error": ""}
        token = get_baidu_ocr_token()  # 获取AI OCR Token
        pages = []

        # -------------------------- 用pdfplumber转图片 --------------------------
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages):
                page_no=page_idx+1
                current_progress = int((page_idx + 1) / total_pages * 100)
                parse_status[pdf_name]["progress"] = current_progress
                time.sleep(0.5)

                # 1. 用pdfplumber将页面转为高清图片（分辨率300DPI，与原逻辑一致）
                img = page.to_image(resolution=300)
                # 转为PIL Image格式，后续AI识别可用
                pil_img = img.original

                # 2. 临时保存图片（供AI识别）
                temp_img_path = f"temp_{pdf_name}_{page_idx + 1}.png"

                # 调用压缩函数（替换直接保存的逻辑，得到缩放后的图片尺寸
                new_size=compress_image(pil_img,temp_img_path)
                print(f"第{page_no}页压缩后尺寸: {new_size}")

                # 3. 后续AI识别逻辑不变（复用原有代码）
                ai_result = ai_extract_text_with_position(temp_img_path, token)
                print(f"第{page_no}页的百度orc提取结果:  ",ai_result)
                # 删除临时图片
                os.remove(temp_img_path)

                # 4. 整理页面数据（关键：映射AI位置到PDF页面坐标）
                pdf_page_width = page.width  # PDF转图后的宽度（与PDF页面比例一致）
                pdf_page_height = page.height
                ai_text = ai_result["full_text"]

                # 5. AI辅助判断文本类型（标题/表格/普通文本，用于相关度排序）
                is_title = any(
                    [len(line) > 5 and len(line) < 50 and line.endswith("模块") for line in ai_result["text_list"]]
                )
                is_table = any(
                    [re.search(r"针脚|端子|序号|功能", line) for line in ai_result["text_list"]]
                )

                pages.append({
                    "page_num": page_idx + 1,
                    "ai_text": ai_text,  # AI提取的完整文字
                    "ai_text_list": ai_result["text_list"],  # 文字列表
                    "ai_positions": ai_result["positions"],  # AI提取的位置信息（含置信度）
                    "is_title_page": is_title,
                    "is_table_page": is_table,
                    "page_width": pdf_page_width,
                    "page_height": pdf_page_height,
                    "page_info": f"第{page_idx + 1}页（AI识别完成）",
                    "image_width":new_size[0],
                    "image_height": new_size[1]
                })
            print(f"该pdf解析数据为：{pages}")

            # 6. 缓存AI解析结果
            pdf_cache[pdf_name] = {
                "total_pages": total_pages,
                "pages": pages,
                "pdf_name": pdf_name,
                "parse_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            parse_status[pdf_name] = {"status": "completed", "progress": 100, "error": ""}

    except Exception as e:
        error_msg = str(e)[:200]
        parse_status[pdf_name] = {"status": "failed", "progress": 0, "error": f"AI解析失败：{error_msg}"}
        print(f"PDF {pdf_name} AI解析错误：{str(e)}")


# -------------------------- 原有接口（仅修改搜索逻辑，其他不变） --------------------------
@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("static/index.html")


@app.get("/pdfs")
async def get_pdf_list():
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
        "note": "本地pdfs文件夹中的PDF文件（全AI解析）"
    }


@app.get("/viewer/{pdf_name}")
async def get_viewer(pdf_name: str):
    return FileResponse("static/viewer.html")


@app.get("/pdf/{pdf_name}")
async def get_pdf_file(pdf_name: str):
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    if not os.path.exists(pdf_path) or not pdf_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=404, detail=f"PDF文件不存在：{pdf_name}")
    return FileResponse(path=pdf_path, filename=pdf_name, media_type="application/pdf")


# -------------------------- AI搜索接口（支持同义词+智能定位） --------------------------
@app.get("/search/{pdf_name}")
async def search_in_pdf(pdf_name: str, query: str):
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    status = parse_status[pdf_name]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"PDF未完成AI解析，当前状态：{status['status']}")
    pdf_data = pdf_cache.get(pdf_name)
    if not pdf_data:
        raise HTTPException(status_code=500, detail="AI解析结果缓存丢失")

    # 1. 同义词扩展（AI搜索优化）
    expanded_queries = expand_synonyms(query)
    results = []

    # 2. 逐页匹配搜索（基于AI提取的文字和位置）
    for page in pdf_data["pages"]:
        page_num = page["page_num"]
        page_text_lower = page["ai_text"].lower()
        pdf_page_width = page["page_width"]  # PDF页面原始宽度（单位：点）
        pdf_page_height = page["page_height"]  # PDF页面原始高度（单位：点）
        image_width = page["image_width"]  # 转图后的像素宽度
        image_height = page["image_height"]  # 转图后的像素高度
        # 检查是否包含任一扩展查询词
        if not any(q in page_text_lower for q in expanded_queries):
            continue

        # 3. 计算相关度分数（按需求：标题>表格>普通文本）
        score = 1.0
        if page["is_title_page"]:
            score += 4.0  # 标题权重最高
        elif page["is_table_page"]:
            score += 2.5  # 表格/描述次之

        # 4. 提取匹配位置（AI返回的位置映射到PDF页面）
        match_positions = []
        for idx, text in enumerate(page["ai_text_list"]):
            if any(q in text.lower() for q in expanded_queries):
                pos = page["ai_positions"][idx]
                # 映射AI位置到PDF.js所需坐标（比例一致）
                match_position = {
                    "x": pos["x0"],
                    "y": pos["y0"],
                    "width": pos["width"],
                    "height": pos["height"],
                    "confidence": pos["confidence"],
                    "page_num": page_num,
                    "match_text": text  # 匹配的文字内容
                }
                match_positions.append(match_position)

        # 5. 生成上下文片段
        snippet = page["ai_text"][:200] if len(page["ai_text"]) > 200 else page["ai_text"]
        snippet = f"...{snippet}..." if len(page["ai_text"]) > 200 else snippet

        results.append({
            "page_num": page_num,
            "score": round(score, 1),
            "position": match_positions,
            "snippet": snippet,
            "is_title": page["is_title_page"],
            "is_table": page["is_table_page"],
            "match_queries": [q for q in expanded_queries if q in page_text_lower],
            "dimensions": {
                "pdf_page_width": pdf_page_width,
                "pdf_page_height": pdf_page_height,
                "image_width": image_width,
                "image_height": image_height
            }

        })

    # 6. 按相关度排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return {
        "target_pdf": pdf_name,
        "original_query": query,
        "expanded_queries": expanded_queries,
        "results": results,
        "total": len(results),
        "total_pages": pdf_data["total_pages"]
    }


 # -------------------------- 新增：AI文档问答接口 --------------------------
@app.get("/qa/{pdf_name}")
async def document_qa(pdf_name: str, question: str):
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    if parse_status[pdf_name]["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"PDF未完成AI解析，无法问答")
    answer = ai_document_qa(pdf_name, question)
    return {
    "target_pdf": pdf_name,
    "question": question,
    "answer": answer
 }


# -------------------------- 原有接口（触发解析、查询状态，无改动） --------------------------
@app.post("/parse-trigger/{pdf_name}")
async def trigger_parse(pdf_name: str, background_tasks: BackgroundTasks):
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    current_status = parse_status[pdf_name]["status"]
    if current_status == "processing":
        return {"success": True, "message": f"AI正在解析：{pdf_name}", "status": current_status,
                "progress": parse_status[pdf_name]["progress"]}
    if current_status == "completed":
        return {"success": True, "message": f"AI已解析完成：{pdf_name}", "status": current_status,
                "parse_time": pdf_cache[pdf_name]["parse_time"]}
    background_tasks.add_task(parse_pdf_async, pdf_name)
    return {"success": True, "message": f"AI解析任务已启动：{pdf_name}", "status": "processing", "progress": 0}


@app.get("/parse-status/{pdf_name}")
async def get_parse_status(pdf_name: str):
    if pdf_name not in parse_status:
        raise HTTPException(status_code=404, detail=f"PDF不存在：{pdf_name}")
    return {
        "target_pdf": pdf_name,
        "status": parse_status[pdf_name]["status"],
        "progress": parse_status[pdf_name]["progress"],
        "error": parse_status[pdf_name]["error"],
        "last_check_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }


# -------------------------- 服务启动 --------------------------
if __name__ == "__main__":
    import uvicorn
    test = expand_synonyms("尿素")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True, timeout_keep_alive=600)