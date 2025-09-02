import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from sklearn.linear_model import LinearRegression
import numpy as np

# Setup page
st.set_page_config(page_title="تحليل ترك العمل", layout="wide")
st.title("📊 تحليل ترك العمل - الموارد البشرية")

# Arabic text fixer
def fix_arabic(text):
    return get_display(reshape(text))

def show_reason_bargraph(filtered):
    """Shows horizontal bar graph of number of people leaving for each reason with clear Arabic labels"""
    reason_counts = filtered['السبب'].value_counts()
    reshaped_labels = [fix_arabic(str(label)) for label in reason_counts.index]

    fig_reason, ax_reason = plt.subplots(figsize=(12, 8))  # Wider for horizontal bars

    bars = ax_reason.barh(range(len(reason_counts)), reason_counts.values, color='lightgreen')

    # Set Arabic labels on y-axis
    ax_reason.set_yticks(range(len(reason_counts)))
    ax_reason.set_yticklabels(reshaped_labels, fontsize=12)

    ax_reason.set_title(fix_arabic("عدد الأشخاص الذين تركوا لكل سبب"), fontsize=14)
    ax_reason.set_xlabel(fix_arabic("عدد الأشخاص"), fontsize=12)
    ax_reason.set_ylabel(fix_arabic("السبب"), fontsize=12)

    # Annotate counts at end of bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax_reason.annotate(f'{int(width)}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(5, 0),
                           textcoords="offset points",
                           ha='left', va='center', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig_reason)

def show_age_group_count(filtered, age_labels, selected_dept):
    """Show age group counts for males and females as a line plot with two lines."""
    grouped = filtered.groupby(['الجنس_mapped', 'فئة العمر']).size().unstack(fill_value=0)

    fig_age, ax_age = plt.subplots(figsize=(10, 6))

    # Plot males if present
    if 'ذكر' in grouped.index:
        ax_age.plot(age_labels, grouped.loc['ذكر', age_labels], marker='o', label=fix_arabic('ذكور'), color='blue')
    else:
        ax_age.plot(age_labels, [0]*len(age_labels), marker='o', label=fix_arabic('ذكور'), color='blue')

    # Plot females if present
    if 'أنثى' in grouped.index:
        ax_age.plot(age_labels, grouped.loc['أنثى', age_labels], marker='o', label=fix_arabic('إناث'), color='red')
    else:
        ax_age.plot(age_labels, [0]*len(age_labels), marker='o', label=fix_arabic('إناث'), color='red')

    ax_age.set_xticks(range(len(age_labels)))
    ax_age.set_xticklabels([fix_arabic(label) for label in age_labels], rotation=45)
    ax_age.set_title(fix_arabic(f"عدد المغادرين حسب الفئة العمرية - قسم {selected_dept}"))
    ax_age.set_xlabel(fix_arabic("الفئة العمرية"))
    ax_age.set_ylabel(fix_arabic("عدد المغادرين"))
    ax_age.grid(True)
    ax_age.legend()

    # Add counts on points for both lines
    for gender, color in [('ذكر', 'blue'), ('أنثى', 'red')]:
        if gender in grouped.index:
            counts = grouped.loc[gender, age_labels]
            for i, v in enumerate(counts):
                ax_age.text(i, v + 0.2, str(v), ha='center', color=color)

    st.pyplot(fig_age)

# File uploader and processing
uploaded_file = st.file_uploader("📎 قم بتحميل ملف Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)

        if "ترك" not in xls.sheet_names:
            st.error("❌ ورقة 'ترك' غير موجودة في الملف.")
            st.stop()

        df_reasons = xls.parse("ترك").dropna(how='all')
        df_reasons.columns = df_reasons.columns.str.strip()

        required_columns = ['الإدارة', 'السبب', 'عدد السنوات', 'الجنس', 'العمر', 'الوضع الاجتماعي']
        missing_cols = [col for col in required_columns if col not in df_reasons.columns]
        if missing_cols:
            st.error(f"❌ الأعمدة التالية مفقودة في ورقة 'ترك': {', '.join(missing_cols)}")
            st.stop()

        departments = df_reasons['الإدارة'].dropna().unique().tolist()
        selected_dept = st.selectbox("🧭 اختر الإدارة", departments)

        if selected_dept:
            filtered = df_reasons[df_reasons['الإدارة'] == selected_dept].copy()

            if filtered.empty:
                st.warning("⚠️ لا توجد بيانات لهذا القسم.")
            else:
                st.subheader("📝 أسباب ترك العمل:")
                show_reason_bargraph(filtered)

                # Word Cloud after reasons bar graph
                text_series = filtered['السبب'].dropna().astype(str)
                stopwords = set(["أسباب", "اسباب"])
                filtered_words = []
                for text in text_series:
                    for word in text.split():
                        if word not in stopwords:
                            filtered_words.append(word)
                text_cleaned = ' '.join(filtered_words)
                if text_cleaned.strip():
                    wordcloud = WordCloud(
                        font_path="Amiri-Regular.ttf",
                        width=800,
                        height=400,
                        background_color='white',
                        collocations=False
                    ).generate(get_display(reshape(text_cleaned)))
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.subheader("☁️ سحابة الكلمات لأسباب ترك العمل")
                    st.pyplot(fig_wc)
                else:
                    st.write("لا توجد بيانات كافية لإنشاء سحابة الكلمات.")

                # Average years of service
                avg_years = pd.to_numeric(filtered['عدد السنوات'], errors='coerce').mean()
                st.write(f"⏳ متوسط سنوات العمل: {avg_years:.2f} سنة" if not pd.isna(avg_years) else "⏳ لا توجد بيانات صالحة.")

                # Prepare gender mapping
                filtered['الجنس'] = filtered['الجنس'].astype(str).str.strip()
                filtered['الجنس_clean'] = filtered['الجنس'].str.lower()
                gender_map = {
                    'male': 'ذكر',
                    'ذكر': 'ذكر',
                    'female': 'أنثى',
                    'أنثى': 'أنثى',
                    'انثى': 'أنثى',
                }
                filtered['الجنس_mapped'] = filtered['الجنس_clean'].map(gender_map)

                # === PREDICTION SECTION ===
                if 'سنة المغادرة' not in filtered.columns:
                    st.warning("⚠️ لا توجد بيانات عن سنة المغادرة للتنبؤ.")
                else:
                    filtered['سنة المغادرة'] = pd.to_numeric(filtered['سنة المغادرة'], errors='coerce')
                    filtered_pred = filtered.dropna(subset=['سنة المغادرة'])

                    if filtered_pred.empty:
                        st.warning("⚠️ لا توجد بيانات صالحة لسنة المغادرة للتنبؤ.")
                    else:
                        min_year = int(filtered_pred['سنة المغادرة'].min())
                        max_year = int(filtered_pred['سنة المغادرة'].max())
                        future_max_year = max_year + 10

                        selected_year = st.slider(
                            fix_arabic("اختر السنة للتنبؤ بعدد المغادرين"),
                            min_value=min_year,
                            max_value=future_max_year,
                            value=max_year
                        )

                        # Aggregate counts by year and gender
                        year_gender_counts = filtered_pred.groupby(['سنة المغادرة', 'الجنس_mapped']).size().unstack(fill_value=0)
                        all_years = list(range(min_year, future_max_year + 1))
                        year_gender_counts = year_gender_counts.reindex(all_years, fill_value=0)

                        def fit_predict(years, counts, pred_year):
                            X = np.array(years).reshape(-1, 1)
                            y = np.array(counts)
                            model = LinearRegression()
                            model.fit(X, y)
                            pred = model.predict(np.array([[pred_year]]))[0]
                            return max(pred, 0)

                        years = all_years
                        male_counts = year_gender_counts.get('ذكر', pd.Series([0]*len(years), index=years))
                        female_counts = year_gender_counts.get('أنثى', pd.Series([0]*len(years), index=years))

                        male_pred = fit_predict(years, male_counts, selected_year)
                        female_pred = fit_predict(years, female_counts, selected_year)

                        # Plot historical + predicted
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
                        ax_pred.plot(years, male_counts, label=fix_arabic('ذكور - تاريخ'), color='blue', marker='o')
                        ax_pred.plot(years, female_counts, label=fix_arabic('إناث - تاريخ'), color='red', marker='o')

                        ax_pred.scatter([selected_year], [male_pred], color='blue', marker='X', s=100,
                                        label=fix_arabic(f'تنبؤ ذكور لعام {selected_year}'))
                        ax_pred.scatter([selected_year], [female_pred], color='red', marker='X', s=100,
                                        label=fix_arabic(f'تنبؤ إناث لعام {selected_year}'))

                        ax_pred.set_title(fix_arabic(f"توقع عدد المغادرين لسنة {selected_year}"))
                        ax_pred.set_xlabel(fix_arabic("السنة"))
                        ax_pred.set_ylabel(fix_arabic("عدد المغادرين"))
                        ax_pred.legend()
                        ax_pred.grid(True)

                        st.pyplot(fig_pred)

                        st.write(f"🔮 التنبؤ بعدد المغادرين في سنة {selected_year}:")
                        st.write(f"- {fix_arabic('ذكور')}: {male_pred:.1f}")
                        st.write(f"- {fix_arabic('إناث')}: {female_pred:.1f}")

                # Gender Distribution Pie Chart
                gender_counts_total = filtered['الجنس_mapped'].value_counts()
                gender_labels_total = [fix_arabic(label) for label in gender_counts_total.index.tolist()]
                gender_sizes_total = gender_counts_total.values

                fig_gender, ax_gender = plt.subplots(figsize=(4, 4))
                if len(gender_sizes_total) > 0:
                    ax_gender.pie(gender_sizes_total, labels=gender_labels_total, autopct='%1.1f%%', startangle=140)
                    ax_gender.set_title(fix_arabic('نسبة الذكور والإناث'))
                    ax_gender.axis('equal')
                else:
                    ax_gender.text(0.5, 0.5, fix_arabic('لا توجد بيانات للجنس'), ha='center', va='center')
                    ax_gender.axis('off')
                st.pyplot(fig_gender)

                # Marital Status Pie Charts
                filtered['الوضع الاجتماعي'] = filtered['الوضع الاجتماعي'].astype(str).str.strip()
                marital_filtered = filtered[filtered['الوضع الاجتماعي'].isin(['متزوج/ة', 'أعزب/عزباء'])]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                # Male marital
                male_data = marital_filtered[marital_filtered['الجنس_mapped'] == 'ذكر']
                male_counts = male_data['الوضع الاجتماعي'].value_counts()
                male_labels = [fix_arabic(label) for label in male_counts.index.tolist()]
                male_sizes = male_counts.values
                if len(male_sizes) > 0:
                    axes[0].pie(male_sizes, labels=male_labels, autopct='%1.1f%%', startangle=90)
                    axes[0].set_title(fix_arabic('حالة الزواج للذكور'))
                    axes[0].axis('equal')
                else:
                    axes[0].text(0.5, 0.5, fix_arabic('لا توجد بيانات للذكور'), ha='center', va='center')
                    axes[0].axis('off')

                # Female marital
                female_data = marital_filtered[marital_filtered['الجنس_mapped'] == 'أنثى']
                female_counts = female_data['الوضع الاجتماعي'].value_counts()
                female_labels = [fix_arabic(label) for label in female_counts.index.tolist()]
                female_sizes = female_counts.values
                if len(female_sizes) > 0:
                    axes[1].pie(female_sizes, labels=female_labels, autopct='%1.1f%%', startangle=90)
                    axes[1].set_title(fix_arabic('حالة الزواج للإناث'))
                    axes[1].axis('equal')
                else:
                    axes[1].text(0.5, 0.5, fix_arabic('لا توجد بيانات للإناث'), ha='center', va='center')
                    axes[1].axis('off')

                st.pyplot(fig)

                  # Age grouping line plot
                age_bins = [20, 30, 40, 50, 60, 70]
                age_labels = ['20–29', '30–39', '40–49', '50–59', '60–69']
                filtered['فئة العمر'] = pd.cut(
                    pd.to_numeric(filtered['العمر'], errors='coerce'),
                    bins=age_bins,
                    labels=age_labels,
                    right=False
                )
                show_age_group_count(filtered, age_labels, selected_dept)

                # 🔮 Prediction snippet with Arabic reshaping applied
                # Ensure 'سنة المغادرة' exists by extracting from 'تاريخ الترك'
                if 'سنة المغادرة' not in filtered.columns:
                    filtered['سنة المغادرة'] = pd.to_datetime(filtered['تاريخ الترك'], errors='coerce').dt.year

                filtered_pred = filtered.dropna(subset=['سنة المغادرة'])
                if filtered_pred.empty:
                    st.warning(fix_arabic("⚠️ لا توجد بيانات صالحة للتنبؤ."))
                else:
                    # Range of years
                    min_year = int(filtered_pred['سنة المغادرة'].min())
                    max_year = int(filtered_pred['سنة المغادرة'].max())
                    future_max_year = max_year + 10

                    # User selects year to predict
                    selected_year = st.slider(
                       "اختر السنة للتنبؤ بعدد المغادرين",
                        min_value=min_year,
                        max_value=future_max_year,
                        value=max_year
                    )

                    # Group data by year and gender
                    year_gender_counts = filtered_pred.groupby(['سنة المغادرة', 'الجنس_mapped']).size().unstack(fill_value=0)
                    all_years = list(range(min_year, future_max_year + 1))
                    year_gender_counts = year_gender_counts.reindex(all_years, fill_value=0)

                    # Function for fitting linear regression and predicting
                    def fit_predict(years, counts, pred_year):
                        X = np.array(years).reshape(-1, 1)
                        y = np.array(counts)
                        model = LinearRegression()
                        model.fit(X, y)
                        pred = model.predict(np.array([[pred_year]]))[0]
                        return max(pred, 0)

                    # Male and Female counts
                    male_counts = year_gender_counts.get('ذكر', pd.Series([0]*len(all_years), index=all_years))
                    female_counts = year_gender_counts.get('أنثى', pd.Series([0]*len(all_years), index=all_years))

                    # Predictions
                    male_pred = fit_predict(all_years, male_counts, selected_year)
                    female_pred = fit_predict(all_years, female_counts, selected_year)

                    # Plot predictions
                    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
                    ax_pred.plot(all_years, male_counts, marker='o', label=fix_arabic('ذكور'), color='blue')
                    ax_pred.plot(all_years, female_counts, marker='o', label=fix_arabic('إناث'), color='red')
                    ax_pred.scatter(selected_year, male_pred, color='blue', marker='x', s=100)
                    ax_pred.scatter(selected_year, female_pred, color='red', marker='x', s=100)

                    ax_pred.set_title(fix_arabic(f"توقع عدد المغادرين في سنة {selected_year}"))
                    ax_pred.set_xlabel(fix_arabic("السنة"))
                    ax_pred.set_ylabel(fix_arabic("عدد المغادرين"))
                    ax_pred.legend()
                    ax_pred.grid(True)
                    st.pyplot(fig_pred)

                    # عرض التوقعات بطريقة نصية واضحة مع معالجة الحروف العربية بشكل كامل
                    pred_text_title = f"🔮 التنبؤ بعدد المغادرين في سنة {selected_year}"
                    pred_text_male = f"ذكور: {male_pred:.1f}"
                    pred_text_female = f"إناث: {female_pred:.1f}"
                  
                    st.write(pred_text_title)
                    st.write(pred_text_male)
                    st.write(pred_text_female)


    except Exception as e:
        st.error("❌ حدث خطأ أثناء معالجة البيانات.")
        st.exception(e)
