import os
import glob
import pandas as pd
import re
from pyvis.network import Network
import html
import json
from dotenv import load_dotenv

# --- 1. 配置加载 ---
load_dotenv()
ANALYSIS_BASE_FOLDER = os.getenv('ANALYSIS_BASE_FOLDER', 'default_analysis_folder')
# 更新：输入文件夹现在是 'data'
DATA_FOLDER = os.path.join(ANALYSIS_BASE_FOLDER, 'data')
OUTPUT_FOLDER = os.path.join(ANALYSIS_BASE_FOLDER, 'output')
ABANDON_CSV_FILE = 'abandon.csv'
SUMMARY_CSV_FILE = 'citation_summary_detailed.csv'
HTML_PREFIX = 'citation_network'

# --- 辅助函数 ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[\W_]+', '', text).lower()

def get_color_palette(num_colors):
    palette = ["#58A6FF", "#3FB950", "#F78166", "#D2A8FF", "#F0883E", "#A371F7", "#79C0FF", "#FFAB70", "#E0BF69", "#54D0C3", "#FF7B72", "#BF87FF", "#FF9EC8", "#33B0FF", "#8DDB81"]
    return [palette[i % len(palette)] for i in range(num_colors)]

def generate_beautified_options(groups_config):
    options = {
        "nodes": {"font": {"color": "#d1d5db", "size": 12, "strokeWidth": 0}, "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 10, "x": 5, "y": 5}},
        "edges": {"color": {"color": "#6b7280", "highlight": "#58A6FF", "hover": "#80c1ff", "inherit": False, "opacity": 0.6}, "smooth": {"type": "cubicBezier", "forceDirection": "vertical", "roundness": 0.8}, "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}}},
        "layout": {"hierarchical": {"enabled": True, "direction": "UD", "sortMethod": "directed", "nodeSpacing": 200, "treeSpacing": 250}},
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "tooltipDelay": 200},
        "physics": {"enabled": True, "solver": "hierarchicalRepulsion", "hierarchicalRepulsion": {"nodeDistance": 200}},
        "groups": groups_config
    }
    return json.dumps(options, indent=4)

# --- 2. 核心功能函数 ---

def load_and_preprocess_data(input_folder, output_folder):
    print(f"--- 步骤 1: 从 '{input_folder}' 加载和预处理数据 ---")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"错误：在输入文件夹 '{input_folder}' 中没有找到任何CSV文件。")

    df_list = [pd.read_csv(file, on_bad_lines='skip') for file in all_files]
    all_papers_df = pd.concat(df_list, ignore_index=True)
    all_papers_df.drop_duplicates(subset=['Title'], inplace=True)
    all_papers_df.reset_index(drop=True, inplace=True)
    print(f"共加载了 {len(all_papers_df)} 篇不重复的文献。")

    all_papers_df['normalized_for_check'] = all_papers_df['Title'].apply(normalize_text)
    processed_titles = list(all_papers_df['normalized_for_check'])
    indices_to_abandon = set()

    for i, title_i in enumerate(processed_titles):
        if i in indices_to_abandon: continue
        for j, title_j in enumerate(processed_titles):
            if i == j: continue
            if title_i and title_j and title_i in title_j and title_i != title_j:
                indices_to_abandon.add(i)
                break
    
    if indices_to_abandon:
        print(f"检测到 {len(indices_to_abandon)} 篇子集标题文献，将被排除。")
        abandoned_df = all_papers_df.loc[list(indices_to_abandon)].drop(columns=['normalized_for_check'])
        main_df = all_papers_df.drop(index=list(indices_to_abandon)).drop(columns=['normalized_for_check'])
        
        abandon_path = os.path.join(output_folder, ABANDON_CSV_FILE)
        abandoned_df.to_csv(abandon_path, index=False, encoding='utf-8-sig')
        print(f"已将被排除的文献保存到: {abandon_path}")
    else:
        print("未检测到作为子集的标题。")
        main_df = all_papers_df.drop(columns=['normalized_for_check'])
        abandoned_df = pd.DataFrame()

    print(f"将使用 {len(main_df)} 篇文献进行引文网络分析。")
    return main_df, pd.DataFrame()

def build_citation_network(main_df):
    print("\n--- 步骤 2: 构建引用网络 ---")
    main_df['normalized_title'] = main_df['Title'].apply(normalize_text)
    
    title_map = {row['normalized_title']: {'original_title': row['Title'], 'year': row['Year'], 'doi': row['DOI']}
                 for _, row in main_df.iterrows()}
    
    citation_edges, in_degree_counts, out_degree_counts = [], {}, {}
    for norm_title in title_map:
        in_degree_counts[norm_title] = 0
        out_degree_counts[norm_title] = 0

    known_normalized_titles = set(title_map.keys())
    for _, citing_paper in main_df.iterrows():
        citing_title_norm = citing_paper['normalized_title']
        references_norm = normalize_text(citing_paper['References'])
        if not citing_title_norm or not references_norm: continue

        for cited_title_norm in known_normalized_titles:
            if cited_title_norm == citing_title_norm: continue
            if cited_title_norm in references_norm:
                citation_edges.append({'source': citing_title_norm, 'target': cited_title_norm})
                in_degree_counts[cited_title_norm] += 1
                out_degree_counts[citing_title_norm] += 1
    
    print(f"找到了 {len(citation_edges)} 条引用关系。")
    return title_map, citation_edges, in_degree_counts, out_degree_counts

def generate_interactive_graphs(
    threshold_pairs, use_and_logic, exclude_isolated_nodes, 
    bg_color, node_size_multiplier, # 新增超参数
    title_map, citation_edges, in_degree_counts, out_degree_counts, output_folder
):
    print("\n--- 步骤 3: 生成交互式网络图 ---")
    
    all_years = sorted(list(set(int(info.get('year', 2000)) for info in title_map.values())))
    color_palette = get_color_palette(len(all_years))
    year_to_color = {year: color for year, color in zip(all_years, color_palette)}
    groups_config = {year: {"shape": "dot", "color": {"background": color, "border": color, "highlight": {"background": "#ffffff", "border": color}, "hover": {"background": "#f0f0f0", "border": color}}} for year, color in year_to_color.items()}

    logic_str = "AND" if use_and_logic else "OR"

    for out_thresh, in_thresh in threshold_pairs:
        print(f"\n正在生成网络图 (逻辑: {logic_str}, 出度 >= {out_thresh}, 入度 >= {in_thresh})...")
        
        candidate_nodes = set()
        for norm_title in title_map:
            in_d = in_degree_counts.get(norm_title, 0)
            out_d = out_degree_counts.get(norm_title, 0)
            passes_degree_filter = (out_d >= out_thresh and in_d >= in_thresh) if use_and_logic else (out_d >= out_thresh or in_d >= in_thresh)
            if passes_degree_filter:
                candidate_nodes.add(norm_title)

        final_nodes_to_draw = set()
        if exclude_isolated_nodes and candidate_nodes:
            subgraph_in_degree = {node: 0 for node in candidate_nodes}
            subgraph_out_degree = {node: 0 for node in candidate_nodes}
            for edge in citation_edges:
                if edge['source'] in candidate_nodes and edge['target'] in candidate_nodes:
                    subgraph_out_degree[edge['source']] += 1
                    subgraph_in_degree[edge['target']] += 1
            
            for node in candidate_nodes:
                if subgraph_in_degree[node] > 0 or subgraph_out_degree[node] > 0:
                    final_nodes_to_draw.add(node)
        else:
            final_nodes_to_draw = candidate_nodes

        if not final_nodes_to_draw:
            print(f"在此条件下，没有文献满足要求，跳过生成此图。"); continue

        print(f"找到 {len(final_nodes_to_draw)} 个满足条件的节点进行绘制。")
        
        net = Network(height='100vh', width='100%', bgcolor=bg_color, font_color='#d1d5db', directed=True)
        
        for norm_title in final_nodes_to_draw:
            info = title_map[norm_title]
            in_d = in_degree_counts.get(norm_title, 0)
            out_d = out_degree_counts.get(norm_title, 0)
            year = int(info.get('year', 2000))
            doi = info.get('doi')
            doi_url = f"https://doi.org/{doi}" if pd.notna(doi) else None
            
            # --- Tooltip彻底修复：使用纯文本和换行符 \n ---
            hover_title = (f"{info['original_title']}\n\n"
                         f"Year: {info.get('year', 'N/A')}\n"
                         f"Cited By (In): {in_d}\n"
                         f"Cites (Out): {out_d}")
            
            # --- 节点大小应用缩放乘数 ---
            node_size = (12 + in_d * 1.5) * node_size_multiplier
            
            net.add_node(n_id=norm_title, label=info['original_title'], value=node_size,
                         title=hover_title, level=year, group=year, doi_url=doi_url)

        for edge in citation_edges:
            if edge['source'] in final_nodes_to_draw and edge['target'] in final_nodes_to_draw:
                net.add_edge(source=edge['source'], to=edge['target'])

        options_str = generate_beautified_options(groups_config)
        net.set_options(options_str)
        
        output_filename = f"{HTML_PREFIX}_logic_{logic_str.lower()}_out_{out_thresh}_in_{in_thresh}.html"
        output_path = os.path.join(output_folder, output_filename)
        net.save_graph(output_path)

        with open(output_path, 'r+', encoding='utf-8') as f:
            content = f.read()
            f.seek(0, 0)
            info_box_html = (f"<div style='position: absolute; top: 15px; left: 15px; padding: 10px; background-color: rgba(40, 40, 40, 0.85); "
                             f"color: #f3f4f6; font-family: sans-serif; font-size: 13px; border-radius: 6px; border: 1px solid #4b5563; max-width: 300px;'>"
                             f"<strong>Filter:</strong><br>Out-Degree ≥ {out_thresh} {logic_str} In-Degree ≥ {in_thresh}"
                             f"</div>")
            js_click_handler = """<script>document.addEventListener("DOMContentLoaded",function(){network&&network.on("click",function(e){if(e.nodes.length>0){var o=nodes.get(e.nodes[0]);o.doi_url&&window.open(o.doi_url,"_blank")}})});</script>"""
            final_html = content.replace('<body>', f'<body>{info_box_html}').replace('</body>', js_click_handler + '</body>')
            f.write(final_html)
        
        print(f"美化后的网络图已保存到: {output_path}")

def generate_summary_csv(title_map, in_degree_counts, out_degree_counts, output_folder):
    print("\n--- 步骤 4: 生成摘要CSV文件 ---")
    table_data = []
    for norm_title, info in title_map.items():
        table_data.append({
            'Year': info.get('year', 'N/A'), 'Title': info['original_title'], 'DOI': info.get('doi', 'N/A'),
            'Local_Cited_By_In_Degree': in_degree_counts.get(norm_title, 0),
            'Local_Cites_Out_Degree': out_degree_counts.get(norm_title, 0),
        })
    
    table_df = pd.DataFrame(table_data)
    table_df['Year'] = pd.to_numeric(table_df['Year'], errors='coerce')
    table_df.sort_values(by='Year', ascending=True, na_position='first', inplace=True)
    output_path = os.path.join(output_folder, SUMMARY_CSV_FILE)
    table_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据表格已成功保存到: {output_path}")

# --- 3. 主函数入口 ---
def main():
    """主执行函数"""
    print(f"--- 开始分析项目: {ANALYSIS_BASE_FOLDER} ---")
    
    # --- 超参数设置 ---
    DEGREE_THRESHOLD_PAIRS = [
        (1, 1), # 包含所有引用至少1次 AND/OR 被引用至少1次的文献
        (0, 1), # 包含所有被引用至少1次的文献（使用AND）
        (2, 2)]  # 可根据需要调整
    USE_AND_LOGIC_FOR_DEGREES = True
    EXCLUDE_ISOLATED_NODES = True
    # 新增超参数
    BACKGROUND_COLOR = '#111827'  # HTML背景色, e.g., '#ffffff' for white
    NODE_SIZE_MULTIPLIER = 1.0     # 节点大小缩放因子, e.g., 1.5 for 150%
    
    try:
        if not os.path.isdir(DATA_FOLDER):
            raise FileNotFoundError(f"输入文件夹不存在: '{DATA_FOLDER}'. 请检查 .env 文件中的 ANALYSIS_BASE_FOLDER 设置。")
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
            print(f"已创建输出文件夹: {OUTPUT_FOLDER}")

        main_df, _ = load_and_preprocess_data(DATA_FOLDER, OUTPUT_FOLDER)
        if main_df.empty:
            print("\n没有可供分析的数据，程序退出。"); return
            
        title_map, edges, in_degrees, out_degrees = build_citation_network(main_df)
        
        generate_interactive_graphs(
            threshold_pairs=DEGREE_THRESHOLD_PAIRS,
            use_and_logic=USE_AND_LOGIC_FOR_DEGREES,
            exclude_isolated_nodes=EXCLUDE_ISOLATED_NODES,
            bg_color=BACKGROUND_COLOR,
            node_size_multiplier=NODE_SIZE_MULTIPLIER,
            title_map=title_map, citation_edges=edges,
            in_degree_counts=in_degrees, out_degree_counts=out_degrees,
            output_folder=OUTPUT_FOLDER
        )
        
        generate_summary_csv(title_map, in_degrees, out_degrees, OUTPUT_FOLDER)
        
        print(f"\n--- 分析项目 '{ANALYSIS_BASE_FOLDER}' 已成功完成！ ---")

    except Exception as e:
        print(f"\n程序执行过程中发生错误: {e}")

if __name__ == "__main__":
    main()
