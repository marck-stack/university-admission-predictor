
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# وظيفة لتحويل النسب المئوية إلى قيم عشرية
def convert_percentage_to_decimal(percentage):
    match = re.match(r"(\d+)%", percentage)
    if match:
        return float(match.group(1)) / 100
    else:
        return None

# وظيفة لتحليل نصوص باستخدام BERT
def predict_with_bert(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    return prediction.item()

# وظيفة لتفاعل المستخدم مع البرنامج
def get_user_input():
    print("مرحبًا بك في محاكاة القبول في الجامعة الأمريكية / جامعة ستانفورد.")
    
    # السؤال عن الدولة
    country = input("من أي دولة تتقدم؟ ")
    
    # السؤال عن المعدل التراكمي
    gpa_input = input("الرجاء إدخال معدلك التراكمي (GPA) من 4.0: ")
    gpa = convert_percentage_to_decimal(gpa_input) * 4.0 if "%" in gpa_input else float(gpa_input)
    
    # السؤال عن درجة SAT
    sat_input = input("الرجاء إدخال درجة SAT (من 1600): ")
    sat_score = int(sat_input)
    
    # السؤال عن درجة ACT
    act_input = input("الرجاء إدخال درجة ACT (من 36): ")
    act_score = int(act_input)
    
    # السؤال عن الأنشطة اللامنهجية
    extracurriculars = input("هل لديك أنشطة لامنهجية (نعم/لا): ").lower()
    extracurriculars_score = 1 if extracurriculars == "نعم" else 0
    
    # السؤال عن نتائج اختبارات اللغة الإنجليزية
    toefl_input = input("هل اجتزت اختبار TOEFL؟ إذا كان نعم، أدخل الدرجة: ")
    toefl_score = int(toefl_input) if toefl_input.lower() != "لا" else 0
    
    # السؤال عن المبالغ المالية المتاحة
    financial_support = input("كم من المال لديك لدعم تعليمك؟ (اكتب المبلغ بالأرقام): ")
    financial_support = float(financial_support)
    
    # إضافة نص حوافز أو بيانات نصية أخرى لتقييمها باستخدام BERT
    motivation_text = input("هل يمكنك تقديم رسالة حوافز أو معلومات إضافية (اختياري): ")
    
    # استدعاء BERT للتنبؤ بالقبول بناءً على النص
    if motivation_text:
        motivation_prediction = predict_with_bert(motivation_text)
        print(f"تنبؤ BERT بالقبول بناءً على النص: {'مقبول' if motivation_prediction == 1 else 'مرفوض'}")
    
    return np.array([gpa, sat_score, act_score, extracurriculars_score, toefl_score, financial_support, country])

# بيانات افتراضية (GPA, SAT, ACT, الأنشطة اللامنهجية, TOEFL, المال المتاح) مع نتائج القبول (1 = مقبول, 0 = مرفوض)
data = {
    'GPA': [3.9, 3.7, 3.8, 4.0, 3.5, 3.6, 3.4, 3.8, 4.0, 3.7],
    'SAT': [1500, 1450, 1600, 1550, 1400, 1480, 1350, 1520, 1600, 1450],
    'ACT': [33, 30, 35, 34, 28, 32, 26, 31, 36, 30],
    'Extracurriculars': [1, 1, 1, 1, 0, 1, 0, 1, 1, 0],
    'TOEFL': [100, 90, 110, 105, 85, 95, 80, 110, 115, 90],
    'Financial Support': [50000, 30000, 70000, 60000, 25000, 35000, 20000, 50000, 80000, 40000],
    'Accepted': [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 1 = مقبول, 0 = مرفوض
}

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data)

# فصل البيانات إلى السمات (X) والهدف (y)
X = df[['GPA', 'SAT', 'ACT', 'Extracurriculars', 'TOEFL', 'Financial Support']]
y = df['Accepted']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تطبيع البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# تدريب نموذج الانحدار اللوجستي (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# تقييم النموذج
accuracy = model.score(X_test_scaled, y_test)
print(f"دقة النموذج على مجموعة الاختبار: {accuracy * 100:.2f}%")

# التفاعل مع المستخدم للحصول على مدخلات
user_input = get_user_input()

# تطبيع المدخلات الخاصة بالمستخدم
user_input_scaled = scaler.transform([user_input[:-1]])  # بدون الدولة في هذه الحالة

# استخدام النموذج للتنبؤ بنتيجة القبول
prediction = model.predict(user_input_scaled)

# إخراج النتيجة للمستخدم
if prediction == 1:
    print("مبروك! تم قبولك في الجامعة.")
else:
    print("للأسف، لم يتم قبولك في الجامعة.")
