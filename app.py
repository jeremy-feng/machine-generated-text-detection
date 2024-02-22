import shap
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from streamlit_shap import st_shap

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizerFast.from_pretrained(
    "JeremyFeng/machine-generated-text-detection"
)
model = BertForSequenceClassification.from_pretrained(
    "JeremyFeng/machine-generated-text-detection"
).to(device)


def pred(x):
    predlist = []
    for text in x:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
            return_attention_mask=True,
        ).to(device)
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        y_score = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().detach()
        predlist.append(y_score)

    predtensor = torch.cat(predlist)
    return predtensor.numpy()


st.title("机器生成文本检测器")

default_texts = [
    "图像识别，是指利用计算机对图像进行处理、分析和理解，以识别各种不同模式的目标和对象的技术，是应用深度学习算法的一种实践应用。现阶段图像识别技术一般分为人脸识别与商品识别，人脸识别主要运用在安全检查、身份核验与移动支付中；商品识别主要运用在商品流通过程中，特别是无人货架、智能零售柜等无人零售领域。图像的传统识别流程分为四个步骤：图像采集→图像预处理→特征提取→图像识别。随着科技的不断进步，图像识别技术也越来越成熟，现阶段已经能够高效准确地处理各种复杂场景。特别是卷积神经网络（CNN）等深度学习模型的运用，使得图像识别的精度大大提升。而随着 5G、云计算和人工智能等新一代信息技术的快速发展，图像识别将有可能在更多领域得到广泛应用，如医疗诊断、自动驾驶、无人机等。而且，有了大数据的支持，我们可以通过更多的样本来训练模型，提高模型的性能。",
    "我们在学习中有学习的环境，你是一个很优秀并且得到过奖学金的人，可见你对学习环境的适应力和掌控力是很强的。现在的状态需要你放下以前的心态，重新来过。从零开始，你要到哪个公司里，现在不招人，那就从侧面能多了解就多了解这个公司的状况和要求，让自己在同行业的可以进入的其他公司里磨练，时刻注视着你要去的地方，按那里的要求来要求自己的日常工作。然后再积累经验，提高自我，逐渐向你理想的公司接近。以上所述，无论你任何时候走入新的工作环境，都需要以谦虚的态度学习，以毅力和耐心去适应。但同时，也要积极向前看，进行自我提升，为未来的职业生涯铺路。通过自我磨练和不断学习，你可以获得新的技巧和知识，进一步理解你想要去的公司的工作方式和要求。在获得这些经验之后，你会发现自己的专业素质和适应能力有了显著的提升，也更接近你的职业目标了。",
    "我今年也大一，处境和你很相似。表面是过得去就行，大学里面还是要保持精神上的独立，如果还未遇到志同道合的同学，建议多和导员还有各科老师沟通，他们都是过来人，会理解你的处境。不要忘记，大学也是锻炼人的社交技巧和团队合作能力的地方。参加一些兴趣社团也是好的选择，可以让你结交到来自不同专业，但有着相同兴趣的人。这样你会发现，原来和你一样迷茫的人其实并不少。这种经历，会让你更加坚定，更懂得如何处理人际关系，如何在艰难困苦中找到自己的方向。同时，也一定要注意自我调节，以保持良好的精神和身体健康。这是你走向成功的重要因素。总的来说，通过这样的方式来发现和解决问题，并随着时间的推移，你会发现自己在很大程度上都有所改变和成长，这是最宝贵的。",
    "本文首先基于 Guo et al. (2023) 整理的中文人类-ChatGPT 问答对比语料集（HC3-Chinese），提取其中的人类生成文本。这些人类生成文本主要有两个来源：一是公开可用的问答数据集，这些数据集中的答案由特定领域的专家给出，或是网络用户投票选出的高质量答案；二是从维基百科和百度百科等资料中构造的“概念 - 解释”问答语句对。",
]

selected_text = st.selectbox(
    "选择一个文本示例或输入待检测文本",
    options=["请选择..."] + default_texts,
)

if selected_text != "请选择...":
    text_area_value = selected_text
else:
    text_area_value = ""

user_input = st.text_area(
    "待检测文本",
    value=text_area_value,
    height=300,
)

if user_input == "":
    st.stop()

y_score = pred([user_input])
if y_score[0] < 0.5:
    st.success(f"该文本是机器生成的概率为 {y_score[0]*100:.2f}%", icon="🧑🏻‍💻")
else:
    st.error(f"该文本是机器生成的概率为 {y_score[0]*100:.2f}%", icon="🤖")

st.subheader("SHAP 分句可解释性分析")
try:
    masker = shap.maskers.Text(tokenizer=r"[\n。.？?！!]")
    explainer = shap.Explainer(pred, masker)
    shap_values = explainer([user_input], fixed_context=1)

    st_shap(shap.plots.text(shap_values, grouping_threshold=0.8), height=300)
except Exception as e:
    if "zero-size array to reduction operation maximum which has no identity" in str(e):
        st.error("❗️文本长度过短，无法使用 SHAP 进行分句可解释性分析")
        st.stop()
    st.exception(e)
