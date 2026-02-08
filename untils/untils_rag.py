# 将检索到的文档中的 page_content 取出组合到一起
import json
import os
from datetime import datetime

from langchain_core.documents import Document


def format_docs_queries(docs:list[Document]) -> str:
    if isinstance(docs, str):
        return docs
    docs = sum(docs,[])# 如果 docs 是 list[list[Document]]，则取出内层的 list[Document]
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_query(docs:list[Document]) -> str:
    if isinstance(docs, str):
        return docs
    return "\n\n".join(doc.page_content for doc in docs)

# 新增：记忆持久化功能（保存到JSON文件）
def save_memory_to_file(memory, session_id="default_user", file_path="conversation_history.json"):
    """保存记忆到JSON文件"""
    # 获取历史对话
    history = memory
    # 转换为可序列化格式
    history_dict = [
        {
            "type": msg["type"],
            "content": msg["content"],
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        } for msg in history
    ]
    # 按用户ID存储（支持多用户）
    data = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    # 如果 session_id 已存在，追加新数据
    if session_id in data:
        data[session_id].extend(history_dict)
    else:
        data[session_id] = history_dict
    # 写入文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_history_from_file( session_id="default_user", file_path="conversation_history.json",k=20):

    """
    k:返回最近k条记录
    从JSON文件加载记忆
    :return list  [role: str, content: str]
    """
    last_x_messages = k * 2  # 每条记录包含用户和AI的消息
    history = []
    if not os.path.exists(file_path):
        return
    # 检查文件是否为空
    if os.path.getsize(file_path) == 0:
        print(f"文件 {file_path} 为空，跳过加载。")
        return
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if session_id not in data:
        return
    # 加载历史对话到记忆
    for msg in data[session_id]:
        if msg["type"] == "human":
            history.append({"human":msg["content"]})
        elif msg["type"] == "ai":
            history.append({"ai":msg["content"]})

    recent_history = history[-last_x_messages:] if len(history) >= last_x_messages else history

    return recent_history

#格式化历史记录
def format_history_for_llm(history):
    history_str = ""
    if isinstance(history, list):
        for i in range(len(history)):
            if "human" in history[i]:  # 如果是用户的问题
                history_str += f"Q: {history[i]['human']}\n"
            elif "ai" in history[i]:  # 如果是 AI 的回答
                history_str += f"A: {history[i]['ai']}\n"
    else:
        history_str = str(history)
    return history_str

#从思考llm的回复构建历史记录
def build_history_from_llm_response(question,response):
    """
    user_message: str 用户消息
    ai_response: str AI回复
    :return list  [role: str, content: str]
    [{"type": "human", "content": "用户消息"}, {"type": "ai", "content": "AI回复"}]
    """
    keyword_index = response.find("\n</think>\n\n")
    new_history = [{"type":"human","content": question}, {"type":"ai","content": response[keyword_index + len("\n</think>\n\n"):]}]

    return new_history

#从无思考llm的回复构建历史记录
def build_history_from_llm_response_no_think(question,response):
    """
    user_message: str 用户消息
    ai_response: str AI回复
    :return list  [role: str, content: str]
    [{"type": "human", "content": "用户消息"}, {"type": "ai", "content": "AI回复"}]
    """
    new_history = [{"type":"human","content": question}, {"type":"ai","content": response}]

    return new_history



if __name__ == "__main__":
    # history = load_history_from_file(session_id="小明", file_path="../数据处理/history_save/conversation_history.json")
    # print(history)
    # print(len(history))
    # history =     [{
    #   "type": "human",
    #   "content": "你能用维基百科查询有关习近平的资料吗",
    # },
    # {
    #   "type": "ai",
    #   "content": "Page: 习近平\nSummary: 习近平（1953年6月15日—），籍贯陕西富平，祖籍河南邓州，生于北京，中国共产党、中华人民共和国政治人物，正国级领导人。其于2012年11月至今任中共中央总书记及中共中央军委主席，2013年3月至今任中华人民共和国主席及国家中央军委主席，是中华人民共和国党、政、军自2012年至今及第五代的最高领导人。\n习近平在北京出生长大，是中华人民共和国开国元勋习仲勋与其第二任夫人齐心的长子，也是首位出生在中华人民共和国成立后的中共最高领导人。习近平在北京接受了小学、初中教育，1969年，因文化大革命对家庭带来的冲击而被迫中止学业，作为知识青年前往陕西省延安市延川县梁家河村参加劳动与工作，在此期间于1974年1月加入中国共产党，并在后期担任了梁家河的村党支部书记。1975年进入清华大学化工系就读，1979年毕业后先后任国务院办公厅及中央军委办公厅秘书。1982年，离京赴河北省正定县先后任县委副书记、书记，开始在地方任职。1985年赴福建省，先后在厦门市、宁德地区、福州市任职，1999年任福建省人民政府省长，成为正部级官员。2002年起，先后任中共浙江省委书记和中共上海市委书记。2007年10月，当选为中共中央政治局常委和中共中央书记处书记，并先后兼任或当选中共中央党校校长、国家副主席、党和国家中央军委副主席等职务。\n2012年11月，在中共十八届一中全会当选为中共中央总书记和中共中央军委主席（后于2017、2022年当选并连任），2013年3月经第十二届全国人民代表大会选举当选为中华人民共和国主席和国家中央军委主席（后于2018、2023年全票当选并获得连任），成为党和国家最高领导人至今。习近平是中华人民共和国首位連任三屆的國家主席，也是至今国家主席任期最長的領導人，任期已達13年。\n习近平是中国共产党第十五届中央候补委员，第十六至二十届中央委员，第十七至二十届中央政治局委员、常委，第十八至二十届中央委员会总书记及第九至十四届全国人大代表。2016年10月，中共十八届六中全会确立了其领导核心的地位，在党内启用“以习近平同志为核心的党中央”这一表述。2017年10月，中共十九大决议将其所提出的习近平新时代中国特色社会主义思想写入《中国共产党章程》。\n习近平第一任妻子是柯玲玲。现任妻子为中国女高音歌唱家彭丽媛，育有一女习明泽。\n\nPage: 对习近平的争议\nSummary: 对习近平的争议，主要包括个人争议、公共政策争议、独裁统治争议以及言论审查争议，此外，中国大陆境外的网络社区和媒体亦有较多的对习近平的负面称呼广为流传，不少针对习近平的抗议时有发生。另外，还存在一些关于习近平其人的谣传。\n自2012年出任中共中央总书记的习近平被普遍认为是中华人民共和国历史上继毛泽东之后最有权势的领导人，并预计会长期执掌政权。他任期内的行为也接二连三产生较大争议，如：掌控并利用反腐敗政策巩固派系以加强對政局稳定、全方面压制香港“反送中”民主运动、中美贸易争端中的决策失误、“战狼式”外交辞令、在2019冠状病毒病疫情初期的隐瞒与轻视其後又倒转走向極端清零政策、2020年中国南方水灾期间的冷处理、言論、法律的嚴厲管制等。由於缺乏专业性的政治团队又加之高度的權力集中決策，部分异议人士戏仿中国政府政治宣传用语，讽刺习近平为“总加速师”，支持其連任，意指加速倒退中共执政的统治基础。2022年10月的中共二十届一中全会上，习近平在面临违背废除干部领导职务终身制的争议中，开始中共中央总书记的第三任期。2023年3月的十四届全国人大一次会议上，习近平在2018年修宪废除任期限制的法律背景下，以2952票赞成、0票反对、0票弃权的结果，全票当选国家主席，自此产生了自1949年中华人民共和国成立以来的首次国家元首三连任。",
    # }]
    # save_memory_to_file(history, session_id="小红", file_path="../数据处理/history_save/conversation_history.json")
    pass
