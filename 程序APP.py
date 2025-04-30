import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "CPR_time": {"type": "numerical", "min": 2.0, "max": 56.0, "default": 30.0},
    "SOFA_score": {"type": "numerical", "min": 0.0, "max": 14.0, "default": 4.0},
    "Initial_cardiac_rhythm": {"type": "categorical", "options": [0, 1]},
    "Bystander_CPR": {"type": "categorical", "options": [0, 1]},
    "sTREM_1": {"type": "numerical", "min": 33.12, "max": 594.09, "default": 200.0},
    "sTREM_2": {"type": "numerical", "min": 38.61, "max": 4721.07, "default": 300.0},
    "NSE": {"type": "numerical", "min": 1.30, "max": 68.52, "default": 10.0},
    "IL_6": {"type": "numerical", "min": 20.27, "max": 181.45, "default": 30.0},
    "Lactate": {"type": "numerical", "min": 0.06, "max": 55.0, "default": 2.0},
    "WBC": {"type": "numerical", "min": 1.57, "max": 156.04, "default": 10.0},
    "hs_TnI": {"type": "numerical", "min": 0.01, "max": 50.87, "default": 0.05},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of Non-survivors is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
