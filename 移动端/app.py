import streamlit as st
import itertools
import random
import math
import time

# ==========================================
# 核心算法引擎 (整合了原 solver.py 和 utils.py)
# ==========================================
def generate_combinations(samples, r):
    return list(itertools.combinations(sorted(samples), r))

def build_coverage_map(candidates, targets, s):
    coverage_map = {}
    target_sets = [(t, set(t)) for t in targets]
    for candidate in candidates:
        candidate_set = set(candidate)
        covered_targets = set()
        for target, target_set in target_sets:
            if len(candidate_set & target_set) >= s:
                covered_targets.add(target)
        coverage_map[candidate] = covered_targets
    return coverage_map

def greedy_set_cover(candidates, targets, coverage_map, randomized=True, top_n=5):
    uncovered_targets = set(targets)
    selected_groups = []
    selected_set = set()
    ordered_candidates = sorted(candidates)

    while uncovered_targets:
        scored_candidates = []
        best_gain = 0
        for candidate in ordered_candidates:
            if candidate in selected_set:
                continue
            gain = len(coverage_map[candidate] & uncovered_targets)
            if gain <= 0:
                continue
            if randomized:
                scored_candidates.append((gain, candidate))
                if gain > best_gain:
                    best_gain = gain
            else:
                if gain > best_gain:
                    best_gain = gain
                    scored_candidates = [(gain, candidate)]
                elif gain == best_gain:
                    scored_candidates.append((gain, candidate))
        
        if not scored_candidates:
            raise RuntimeError("无法覆盖所有目标，请检查参数组合。")

        if randomized:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            top_candidates = scored_candidates[:max(1, top_n)]
            _, chosen_candidate = random.choice(top_candidates)
        else:
            scored_candidates.sort(key=lambda item: item[1])
            _, chosen_candidate = scored_candidates[0]

        selected_groups.append(chosen_candidate)
        selected_set.add(chosen_candidate)
        uncovered_targets -= coverage_map[chosen_candidate]
    return selected_groups

def check_all_targets_covered(selected_groups, targets, s):
    selected_group_sets = [set(g) for g in selected_groups]
    for target in targets:
        target_set = set(target)
        if not any(len(target_set & g_set) >= s for g_set in selected_group_sets):
            return False
    return True

def optimize_by_removing_redundant_groups(selected_groups, targets, s):
    optimized_groups = list(selected_groups)
    changed = True
    while changed:
        changed = False
        for group in list(optimized_groups):
            candidate_groups = [g for g in optimized_groups if g != group]
            if candidate_groups and check_all_targets_covered(candidate_groups, targets, s):
                optimized_groups = candidate_groups
                changed = True
                break
    return sorted(optimized_groups)

def solve(samples, k, j, s, runs=3):
    started_at = time.perf_counter()
    candidates = generate_combinations(samples, k)
    targets = generate_combinations(samples, j)
    coverage_map = build_coverage_map(candidates, targets, s)

    best_result = None
    
    # 增加进度条提示
    progress_bar = st.progress(0)
    for run_index in range(1, runs + 1):
        raw_result = greedy_set_cover(candidates, targets, coverage_map, randomized=True)
        optimized_result = optimize_by_removing_redundant_groups(raw_result, targets, s)
        
        if best_result is None or len(optimized_result) < len(best_result):
            best_result = optimized_result
            
        progress_bar.progress(run_index / runs)
    progress_bar.empty() # 计算完成后清空进度条

    elapsed_seconds = time.perf_counter() - started_at
    stats = {
        "candidate_count": len(candidates),
        "target_count": len(targets),
        "final_result_count": len(best_result),
        "elapsed_seconds": elapsed_seconds
    }
    return best_result, stats


# ==========================================
# Streamlit 前端交互界面
# ==========================================
st.set_page_config(page_title="最优样本选择系统", layout="centered")

st.title("🎯 最优样本选择系统 (Web版)")
st.markdown("基于 **GRASP (贪心随机自适应搜索)** 与逆向剪枝优化的组合设计工具。")

# 侧边栏：参数输入
st.sidebar.header("🛠️ 参数设置")
m = st.sidebar.number_input("总体样本数 m (45-54)", 45, 54, 45)
n = st.sidebar.number_input("选择样本数 n (7-25)", 7, 25, 9)
k = st.sidebar.number_input("小组容量 k (4-7)", 4, 7, 6)
j = st.sidebar.number_input("覆盖参考值 j (s ≤ j ≤ k)", 3, k, 4)
s = st.sidebar.number_input("匹配要求 s (3-7)", 3, j, 4)
runs = st.sidebar.slider("算法迭代次数 (寻找更优解)", 1, 10, 3)

# 主界面：样本输入模式
st.subheader("1. 确定初始样本池")
input_mode = st.radio("选择样本产生方式", ["🎲 随机生成", "✍️ 手动输入"], horizontal=True)

if "🎲 随机生成" in input_mode:
    if st.button("生成随机样本"):
        st.session_state['samples'] = sorted(random.sample(range(1, m + 1), n))
else:
    user_input = st.text_input(f"请输入 {n} 个数字（用空格分隔，范围 1-{m}）：")
    if user_input:
        try:
            parsed_samples = sorted(list(set([int(x) for x in user_input.split()])))
            if len(parsed_samples) == n and all(1 <= x <= m for x in parsed_samples):
                st.session_state['samples'] = parsed_samples
                st.success("样本解析成功！")
            else:
                st.error(f"请输入恰好 {n} 个不重复的数字，且范围在 1 到 {m} 之间。")
        except ValueError:
            st.error("输入格式有误，请确保输入的是纯数字。")

# 执行算法与展示结果
if 'samples' in st.session_state:
    st.info(f"**当前选定样本池 (共 {n} 个):** \n {st.session_state['samples']}")
    
    st.subheader("2. 运行优化算法")
    if st.button("🚀 开始执行最优选择", use_container_width=True):
        
        # 预估复杂度预警 (防止浏览器卡死)
        candidate_count = math.comb(n, k)
        if candidate_count > 100000:
            st.warning("⚠️ 当前参数组合空间极大，计算可能需要较长时间，请耐心等待...")
            
        with st.spinner("🧠 启发式算法计算中，正在寻找全局最优解..."):
            try:
                results, stats = solve(st.session_state['samples'], k, j, s, runs=runs)
                st.success(f"✅ 计算完成！耗时: {stats['elapsed_seconds']:.3f} 秒")
                
                # 结果统计卡片展示
                col1, col2, col3 = st.columns(3)
                col1.metric("候选组合总数", stats['candidate_count'])
                col2.metric("需要覆盖的目标数", stats['target_count'])
                col3.metric("⭐ 最终精简组数", stats['final_result_count'])
                
                # 结果展示与文本生成
                st.markdown("### 📊 最优组合结果")
                result_text = ""
                
                # 用两列的格式整齐地展示结果
                res_col1, res_col2 = st.columns(2)
                for idx, res in enumerate(results):
                    line = f"**组 {idx+1}:** {res}"
                    if idx % 2 == 0:
                        res_col1.markdown(line)
                    else:
                        res_col2.markdown(line)
                    
                    # 按照之前 storage.py 的要求格式化文本，用于下载
                    result_text += f"{idx+1}. {','.join(map(str, res))}\n"
                
                # 生成符合作业规范的 DB txt 文件供下载
                file_name = f"{m}-{n}-{k}-{j}-{s}-1-{len(results)}.txt"
                st.download_button(
                    label="📂 下载标准化 DB 结果文件",
                    data=result_text,
                    file_name=file_name,
                    mime="text/plain",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"计算过程中发生错误: {str(e)}")
