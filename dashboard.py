import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="ABP Estimation", layout="centered")
st.title("💓 ABP Estimation Dashboard")
st.markdown("أدخل إشارات PPG و ECG (كل واحدة 250 نقطة) أو ارفع ملف JSON للتوقّع.")

# الإدخال اليدوي
ppg_input = st.text_area("🔴 إشارة PPG (comma-separated)", "")
ecg_input = st.text_area("🔵 إشارة ECG (comma-separated)", "")

# رفع ملف JSON
uploaded_file = st.file_uploader("📁 أو ارفع ملف JSON", type="json")

# متغير للنتيجة
predicted_abp = None

# زر التوقع
if st.button("🔍 توقّع"):
    try:
        # تحضير البيانات
        if uploaded_file is not None:
            data = json.load(uploaded_file)
        else:
            ppg_list = [float(x.strip()) for x in ppg_input.split(",") if x.strip()]
            ecg_list = [float(x.strip()) for x in ecg_input.split(",") if x.strip()]
            data = {"ppg": ppg_list, "ecg": ecg_list}

        # التحقق من الطول
        if len(data["ppg"]) != 250 or len(data["ecg"]) != 250:
            st.error("❌ يجب أن تحتوي كل إشارة على 250 نقطة بالضبط.")
        else:
            # إرسال الطلب
            response = requests.post(
                "https://web-production-07fc.up.railway.app/predict",
                json=data
            )
            result = response.json()
            st.write("📤 الناتج الخام من النموذج:", result)

            predicted_abp = result.get("predicted_abp", [[]])[0]

            if not predicted_abp:
                st.warning("⚠️ النموذج لم يرجع بيانات ABP.")
            else:
                st.success("✅ تم التوقّع بنجاح!")

                # الرسم البياني
                st.subheader("📊 الرسم البياني للـ ABP:")
                fig, ax = plt.subplots()
                if len(predicted_abp) == 1:
                    ax.plot([0], [predicted_abp[0]], marker='o', color="purple")
                    ax.set_xlim(-1, 1)
                else:
                    ax.plot(predicted_abp, color="purple")

                ax.set_xlabel("Time Frame")
                ax.set_ylabel("ABP Value")
                ax.grid(True)
                st.pyplot(fig)

                # حفظ الملف
                result_json = {"predicted_abp": [predicted_abp]}
                with open("predicted_abp_result.json", "w") as f:
                    json.dump(result_json, f, indent=4)

                # زر تحميل
                with open("predicted_abp_result.json", "rb") as f:
                    st.download_button(
                        label="⬇️ تحميل النتيجة كملف JSON",
                        data=f,
                        file_name="predicted_abp_result.json",
                        mime="application/json"
                    )

    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء المعالجة: {e}")
