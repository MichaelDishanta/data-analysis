import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import jieba

# 设置页面标题和主题
st.set_page_config(page_title="招聘数据可视化系统", page_icon="📊", layout="wide")

# 页面样式
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;  /* 页面背景颜色 */
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        h2 {
            color: #333333;
            margin-top: 20px;
            text-align: center;
        }
        .uploadedFile {
            text-align: center;
            margin: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: None;
            padding: 10px 20px;
            border-radius: 5px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
    """, unsafe_allow_html=True)

# 应用程序标题
st.title("📊 招聘数据可视化系统")

# 页面描述
st.markdown("<h2>通过可视化分析提升招聘数据的洞察力</h2>", unsafe_allow_html=True)

# 创建数据清洗的选项卡
tab1, tab2, tab3 = st.tabs(["数据清洗", "数据分析", "数据可视化"])


# 清洗函数：保留汉字、数字和英文字母，去除其他字符
def clean_text(text):
    cleaned_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 清除非汉字、数字和英文字母的字符
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # 去除多余的空格
    return cleaned_text
# 定义标准化薪资的函数
def standardize_salary(salary):
    salary = str(salary)
    # 将 '万' 转换为 'k'
    if '万' in salary:
        salary = re.sub(r'(\d+(\.\d+)?)万', lambda x: str(int(float(x.group(1)) * 10)) + 'k', salary)
    # 将 '千' 转换为 'k'
    if '千' in salary:
        salary = salary.replace('千', 'k')
    return salary

# 去掉薪资列中的 '·13薪' 之类的部分
def remove_bonus(salary):
    salary = str(salary)
    salary = re.sub(r'\s*·\d+薪', '', salary)
    return salary

# 计算平均薪资
def calculate_average_salary(salary):
    salary = str(salary)
    if '-' in salary:
        parts = salary.split('-')
        if len(parts) == 2:
            try:
                lower = float(parts[0].replace('k', ''))
                upper = float(parts[1].replace('k', ''))
                return f"{(lower + upper) / 2}k"
            except ValueError:
                return salary
    try:
        return f"{float(salary.replace('k', ''))}k"
    except ValueError:
        return salary

# 数据清洗部分
with tab1:
    st.header("🧹 数据清洗")
    col1, col2 = st.columns(2)

    # 数据清洗——挖掘技能关联规则
    with col1:
        st.header("挖掘技能关联规则")
        uploaded_file_cleaning = st.file_uploader("上传需要清洗的CSV文件", type=["csv"], key="cleaning_uploader")

        if uploaded_file_cleaning is not None:
            df_clean = pd.read_csv(uploaded_file_cleaning)
            st.write("原始数据预览:", df_clean.head())
            # 1. 数据去重和去除非法字符整合
            if st.checkbox("第一步：数据去重与去除非法字符"):
                st.write("请选择需要基于哪些字段进行去重：")
                key_columns = st.multiselect("选择去重的关键字段", df_clean.columns)

                if key_columns:
                    # 提供执行按钮
                    if st.button("执行去重和去除非法字符"):
                        # 执行去重
                        original_length = len(df_clean)
                        df_clean = df_clean.drop_duplicates(subset=key_columns)
                        removed_count = original_length - len(df_clean)
                        st.write(f"去重完成，共去除 {removed_count} 条重复数据。")

                        # 执行去除非法字符
                        if '职位描述' in df_clean.columns:
                            descriptions = df_clean['职位描述'].dropna()
                            cleaned_descriptions = descriptions.apply(clean_text)
                            df_clean['清洗后的职位描述'] = cleaned_descriptions
                            st.write("清洗后的职位描述预览:", df_clean[['职位描述', '清洗后的职位描述']].head())

                            # 将所有的技能描述写入到TXT文件
                            skills_list = cleaned_descriptions.tolist()  # 转为列表
                            output_file = "技能描述.txt"  # 定义输出文件名

                            # 将技能描述写入内存中的字符串
                            skills_text = "\n".join([f"[{skill}]" for skill in skills_list])

                            # 提供下载清洗后的数据
                            # cleaned_file_after = df_clean.to_csv(index=False).encode('utf-8')
                            # st.download_button("下载清洗后的数据", cleaned_file_after, "cleaned_data_after.csv",
                            #                    "text/csv")

                            # 提供下载技能描述的TXT文件
                            st.download_button("下载技能描述", skills_text, "技能描述.txt", "text/plain")

                        else:
                            st.warning("数据中不包含'职位描述'列。")
                else:
                    st.warning("请至少选择一个字段进行去重。")

            # 2. 统一技能名称
            if st.checkbox("第二步：统一职位描述的技能名称大小写"):
                # 允许用户上传第一步清洗后的TXT文件
                uploaded_file_previous_cleaned_txt = st.file_uploader("上传去重后的TXT文件", type=["txt"],
                                                                      key="previous_cleaned_txt_uploader")

                if uploaded_file_previous_cleaned_txt is not None:
                    # 读取TXT文件内容
                    skills_list = uploaded_file_previous_cleaned_txt.read().decode('utf-8').strip().splitlines()

                    # 统一技能名称为小写
                    unified_skills = [skill.lower() for skill in skills_list]

                    # 显示统一后的技能名称预览
                    st.write("统一后的职位描述预览:", unified_skills[:10])  # 只显示前10个

                    # 将统一后的技能名称保存为TXT文件
                    unified_skills_text = "\n".join([f"[{skill}]" for skill in unified_skills])

                    # 提供下载统一后的数据
                    st.download_button("下载统一后的技能描述", unified_skills_text, "unified_skills.txt", "text/plain")

            # 3. tdf-idf词频提取
            if st.checkbox("第三步：计算词频并显示前80个词"):
                uploaded_file_previous_cleaned_txt = st.file_uploader("上传统一技能名称的TXT文件", type=["txt"],
                                                                      key="previous_cleaned_txt_uploader2")

                if uploaded_file_previous_cleaned_txt is not None:
                    # 读取TXT文件内容
                    skills_list = uploaded_file_previous_cleaned_txt.read().decode('utf-8').strip().splitlines()

                    # 使用每行的技能词作为输入进行TF-IDF处理
                    # 假设每行都是一个技能描述
                    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                    X = vectorizer.fit_transform(skills_list)

                    # 提取特征名称（词汇）
                    feature_names = vectorizer.get_feature_names_out()

                    # 创建一个 DataFrame，展示每个词汇的 TF-IDF 分数
                    df_tfidf = pd.DataFrame(X.T.toarray(), index=feature_names)


                    # 过滤掉包含中文字符的词汇
                    def contains_chinese(text):
                        return bool(re.search(r'[\u4e00-\u9fff]', text))


                    # 只保留不包含中文字符的词汇
                    df_tfidf = df_tfidf[~df_tfidf.index.map(contains_chinese)]

                    # 计算每个词汇的平均 TF-IDF 分数
                    df_tfidf['mean_score'] = df_tfidf.mean(axis=1)

                    # 按照平均 TF-IDF 分数降序排序
                    df_tfidf = df_tfidf.sort_values(by='mean_score', ascending=False)

                    # 显示前80个最重要的词汇
                    st.write("TF-IDF 分数最高的前 80 个词汇：")
                    st.dataframe(df_tfidf[['mean_score']].head(80))  # 只显示前80行
                    st.write("请自行整理出排名靠前的技能词汇，为下一步分词做准备")

                    # 4. 分词处理
            if st.checkbox("第四步：提取技能词汇并使用自定义分词"):
                # 用户自定义技能词汇表
                custom_skills_input = st.text_area(
                    "请输入自定义技能词汇（以逗号分隔）:",
                    value=""
                )

                # 处理自定义技能词汇
                custom_skills = [skill.strip().lower() for skill in custom_skills_input.split(",")]

                # 将技能词汇加入到 jieba 分词词典中
                for skill in custom_skills:
                    jieba.add_word(skill)


                # 定义分词并提取技能词汇的函数
                def extract_skills(text):
                    # 处理组合词
                    text = re.sub(r'spring\s+boot', 'springboot', text, flags=re.IGNORECASE)
                    text = re.sub(r'spring\s+mvc', 'springmvc', text, flags=re.IGNORECASE)
                    text = re.sub(r'spring\s+cloud', 'springcloud', text, flags=re.IGNORECASE)
                    text = re.sub(r'\bjs\b', 'JavaScript', text, flags=re.IGNORECASE)

                    # 使用 jieba 分词
                    words = jieba.lcut(text)
                    extracted_skills = [word.lower() for word in words if word.lower() in custom_skills]

                    # 去除重复词汇
                    unique_skills = list(set(extracted_skills))

                    return unique_skills


                # 上传文件
                uploaded_file = st.file_uploader("上传文本文件", type="txt")

                if uploaded_file is not None:
                    content = uploaded_file.read().decode("utf-8").splitlines()
                    all_extracted_skills = []

                    for line in content:
                        skills_extracted = extract_skills(line)
                        all_extracted_skills.append(skills_extracted)

                    # 将提取的技能写入新的文本文件
                    output_file_content = "\n".join(
                        [f"[{', '.join(skills)}]" for skills in all_extracted_skills if skills])

                    # 提供下载链接
                    st.download_button(
                        label="下载提取的技能词汇",
                        data=output_file_content,
                        file_name="extracted_skills.txt",
                        mime="text/plain"
                    )
                    st.success("至此清洗步骤结束，请移至数据分析模块")




















        else:
            st.info("请上传需要清洗的CSV文件。")



    # 数据清洗——可视化学历，城市，薪资的关系
    with col2:
        st.header("可视化学历、城市与薪资的关系")
        uploaded_file = st.file_uploader("上传包含薪资数据的CSV文件", type=["csv"])

        if uploaded_file is not None:
            # 读取 CSV 文件
            df = pd.read_csv(uploaded_file)

            if '薪资' in df.columns:
                # 标准化薪资列
                df['薪资'] = df['薪资'].apply(standardize_salary)
                df['薪资'] = df['薪资'].apply(remove_bonus)
                df['薪资'] = df['薪资'].apply(calculate_average_salary)

                # 删除非标准化薪资（不包含 'k' 的薪资）
                df = df[df['薪资'].str.contains(r'\d+k$', na=False)]

                # 显示处理后的数据
                st.write("处理后的薪资数据预览：")
                st.dataframe(df)

                # 转换为 CSV 格式
                csv_data = df.to_csv(index=False).encode('utf-8')

                # 下载按钮，允许用户下载处理后的 CSV 文件
                st.download_button(
                    label="下载处理后的数据",
                    data=csv_data,
                    file_name="processed_salary_data.csv",
                    mime="text/csv"
                )
            else:
                st.warning("CSV 文件中未找到 '薪资' 列。")
        else:
            st.info("请上传包含薪资数据的CSV文件。")

# 数据分析部分
import streamlit as st
import pandas as pd
import re
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# 创建 tab2
with tab2:
    # 设置页面标题
    st.header(" 关联规则分析")

    # 文件上传
    uploaded_file = st.file_uploader("上传包含技能词汇的 TXT 文件", type=["txt"])

    # 支持度和置信度滑动条
    min_support = st.slider("选择最小支持度（Support）", 0.01, 1.0, 0.2, step=0.01)
    min_confidence = st.slider("选择最小置信度（Confidence）", 0.01, 1.0, 0.75, step=0.01)

    # 选择算法：Apriori 或 FP-Growth
    algorithm = st.selectbox("选择关联规则挖掘算法", ["Apriori", "FP-Growth"])

    if uploaded_file is not None:
        # 读取文件内容
        lines = uploaded_file.read().decode('utf-8').splitlines()

        # 自定义解析每一行的技能列表
        def parse_line(line):
            cleaned_line = line.strip().strip('[]')
            skills = [skill.strip() for skill in cleaned_line.split(',')]
            return skills

        # 使用自定义解析函数处理每一行
        transactions = [parse_line(line) for line in lines]

        # 转换为 DataFrame，创建 one-hot 编码
        all_skills = sorted(set(skill for sublist in transactions for skill in sublist))
        onehot_encoded = [{skill: (skill in transaction) for skill in all_skills} for transaction in transactions]
        df = pd.DataFrame(onehot_encoded)

        # 显示 One-Hot 编码的数据预览
        st.write("One-Hot 编码后的数据：")
        st.dataframe(df.head())

        # 根据用户选择的算法运行分析
        if st.button("运行算法分析"):
            with st.spinner(f'正在计算频繁项集和关联规则 ({algorithm})...'):
                # 根据选择的算法计算频繁项集
                if algorithm == "Apriori":
                    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
                else:
                    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

                st.write(f"频繁项集 ({algorithm})：")
                st.dataframe(frequent_itemsets)

                # 生成关联规则
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                # 只保留三项及以上的规则
                three_or_more_item_rules = rules[rules['antecedents'].apply(lambda x: len(x) > 2)]

                # 避免 SettingWithCopyWarning
                three_or_more_item_rules = three_or_more_item_rules.copy()
                three_or_more_item_rules.loc[:, 'antecedents'] = three_or_more_item_rules['antecedents'].apply(
                    lambda x: ', '.join(x))
                three_or_more_item_rules.loc[:, 'consequents'] = three_or_more_item_rules['consequents'].apply(
                    lambda x: ', '.join(x))

                st.write(f"三项及以上的关联规则 ({algorithm})：")
                st.dataframe(three_or_more_item_rules)

                # 下载按钮：频繁项集
                frequent_itemsets_csv = frequent_itemsets.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"下载频繁项集 ({algorithm}) CSV",
                    data=frequent_itemsets_csv,
                    file_name=f"frequent_itemsets_{algorithm.lower()}.csv",
                    mime="text/csv"
                )

                # 下载按钮：关联规则
                rules_csv = three_or_more_item_rules.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"下载关联规则 ({algorithm}) CSV",
                    data=rules_csv,
                    file_name=f"association_rules_{algorithm.lower()}.csv",
                    mime="text/csv"
                )
    else:
        st.info("请上传包含技能词汇的 TXT 文件。")


# 数据可视化部分
with tab3:
    # 第一行: 词云和热力图
    col1, col2 = st.columns(2)

    # 第一列 - 词云图展示
    with col1:
        st.header("🔍 词云图展示")
        uploaded_file_wordcloud = st.file_uploader("上传词云图CSV文件", type=["csv"], key="wordcloud_uploader_1")
        if uploaded_file_wordcloud is not None:
            with st.spinner('正在生成词云图...'):
                df_wordcloud = pd.read_csv(uploaded_file_wordcloud)
                skills = []
                for index, row in df_wordcloud.iterrows():
                    skill_set = eval(row['itemsets'])  # 假设 'itemsets' 列为 frozen set
                    skills.extend(skill_set)
                skill_freq = pd.Series(skills).value_counts()

                # 创建词云
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                    skill_freq)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
        else:
            st.info("请上传CSV文件以生成词云图。")

    # 第二列 - 热力图展示
    with col2:
        st.header("🔥 关联规则热力图展示")
        uploaded_file_heatmap = st.file_uploader("上传热力图CSV文件", type=["csv"], key="heatmap_uploader_1")
        if uploaded_file_heatmap is not None:
            with st.spinner('正在生成热力图...'):
                df_heatmap = pd.read_csv(uploaded_file_heatmap)
                heatmap_data = df_heatmap.pivot_table(index='antecedents', columns='consequents', values='confidence',
                                                      fill_value=0)

                plt.figure(figsize=(20, 12))
                sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu', cbar_kws={'label': 'Confidence'})
                plt.title('关联规则热力图', fontsize=20)
                st.pyplot(plt)
        else:
            st.info("请上传CSV文件以生成热力图。")

    # 第二行: 城市工资柱形图和学历薪资折线图
    col3, col4 = st.columns(2)

    # 城市工资柱形图展示
    with col3:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        st.header("🏙️ 各城市工资柱形图")
        uploaded_file_salary = st.file_uploader("上传城市工资数据CSV文件", type=["csv"], key="salary_uploader_1")
        if uploaded_file_salary is not None:
            with st.spinner('正在生成城市工资柱形图...'):
                salary_data = pd.read_csv(uploaded_file_salary)
                salary_data['City'] = salary_data['所在城市'].str[:2]
                salary_data['Salary'] = salary_data['薪资'].str.extract(r'(\d+\.?\d*)').astype(float)
                city_salary_mean = salary_data.groupby('City')['Salary'].mean().reset_index()

                plt.figure(figsize=(10, 6))
                plt.bar(city_salary_mean['City'], city_salary_mean['Salary'], color='blue', alpha=0.7)
                plt.title('各城市平均工资对比', fontsize=20)
                plt.xlabel('城市', fontsize=14)
                plt.ylabel('平均工资 (K)', fontsize=14)
                plt.xticks(rotation=45)
                st.pyplot(plt)
        else:
            st.info("请上传CSV文件以生成城市工资柱形图。")

    # 学历薪资折线图展示
    with col4:
        st.header("📈 学历与薪资变化折线图")
        uploaded_file_line = st.file_uploader("上传数据CSV文件", type=["csv"], key="line_uploader_1")

        if uploaded_file_line is not None:
            with st.spinner('正在生成折线图...'):
                df_line = pd.read_csv(uploaded_file_line)
                df_line['薪资'] = df_line['薪资'].str.replace('k', '').astype(float)
                education_order = {'大专': 1, '本科': 2, '硕士': 3, '博士': 4}
                df_line['学历顺序'] = df_line['学历'].map(education_order)
                df_grouped = df_line.groupby('学历顺序')['薪资'].mean().reset_index()
                df_grouped['学历'] = df_grouped['学历顺序'].map({v: k for k, v in education_order.items()})

                plt.figure(figsize=(10, 6))
                plt.plot(df_grouped['学历'], df_grouped['薪资'], marker='o', color='orange')
                plt.title('学历与平均薪资变化', fontsize=20)
                plt.xlabel('学历', fontsize=14)
                plt.ylabel('平均薪资 (K)', fontsize=14)
                plt.grid(True)
                st.pyplot(plt)
        else:
            st.info("请上传CSV文件以生成折线图。")

# 底部版权信息
st.markdown("© 2024 招聘数据可视化系统 - All rights reserved.", unsafe_allow_html=True)
