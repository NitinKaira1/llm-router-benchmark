

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LLM Router Benchmark", layout="wide")

RESULTS_PATH = "results/benchmark_results.json"

st.title("🧭 LLM Router: Cost vs. Quality Benchmark")
st.caption("Always-largest-model baseline (Qwen3 8B) vs. a complexity-based router across 4 local Qwen models, on the same 36 prompts.")

try:
    with open(RESULTS_PATH) as f:
        data = json.load(f)
except FileNotFoundError:
    st.error(f"No results found at `{RESULTS_PATH}`. Run `python benchmark.py` first.")
    st.stop()

summary = data["summary"]
results = data["results"]
df = pd.DataFrame(results)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Cost saved*",
    f"{summary['cost_saved_pct']:.1f}%" if summary["cost_saved_pct"] is not None else "N/A",
    f"${summary['total_baseline_cost_usd']:.4f} → ${summary['total_router_cost_usd']:.4f}",
)
col2.metric(
    "Latency saved",
    f"{summary['latency_saved_pct']:.1f}%" if summary["latency_saved_pct"] is not None else "N/A",
    f"{summary['total_baseline_latency_s']:.1f}s → {summary['total_router_latency_s']:.1f}s",
)
col3.metric(
    "Avg quality (1-5)",
    f"{summary['avg_router_quality']:.2f}" if summary["avg_router_quality"] else "N/A",
    f"baseline {summary['avg_baseline_quality']:.2f}" if summary["avg_baseline_quality"] else "",
    delta_color="off",
)
col4.metric(
    "Classifier accuracy",
    f"{summary['classifier_accuracy']*100:.1f}%" if summary["classifier_accuracy"] is not None else "N/A",
)
st.caption("*Cost is reference cloud-equivalent pricing by model size, not actual spend — these models run locally for $0. See README.")

st.divider()

if summary["cost_saved_pct"] is not None and summary["quality_delta"] is not None:
    st.subheader("Takeaway")
    st.markdown(
        f"> Routing cut equivalent API cost by **{summary['cost_saved_pct']:.0f}%** and wall-clock time by "
        f"**{summary['latency_saved_pct']:.0f}%**, with a quality change of **{summary['quality_delta']:+.2f} points** "
        f"(1-5 scale) across {summary['n_prompts']} prompts, running entirely on local hardware."
    )

st.divider()


c1, c2 = st.columns(2)

with c1:
    st.subheader("Where the router sent each query")
    tier_counts = df["router"].apply(lambda r: r["model_tier"]).value_counts().reset_index()
    tier_counts.columns = ["tier", "count"]
    fig = px.pie(tier_counts, names="tier", values="count", hole=0.45,
                 color="tier",
                 category_orders={"tier": ["tiny", "small", "mid", "large"]})
    fig.update_traces(textinfo="label+percent")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Classifier: predicted vs. expected type")
    conf_df = pd.DataFrame({
        "expected": df["expected_label"],
        "predicted": df["router"].apply(lambda r: r["predicted_label"]),
    })
    conf_matrix = pd.crosstab(conf_df["expected"], conf_df["predicted"])
    fig2 = px.imshow(conf_matrix, text_auto=True, color_continuous_scale="Blues",
                      labels=dict(x="Predicted", y="Expected", color="Count"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()


st.subheader("Cost and latency: baseline vs. router")

agg = pd.DataFrame({
    "Path": ["Baseline (always Qwen3 8B)", "Router"],
    "Total cost (ref. $)": [summary["total_baseline_cost_usd"], summary["total_router_cost_usd"]],
    "Total latency (s)": [summary["total_baseline_latency_s"], summary["total_router_latency_s"]],
})

c3, c4 = st.columns(2)
with c3:
    fig3 = px.bar(agg, x="Path", y="Total cost (ref. $)", color="Path",
                  color_discrete_sequence=["#94a3b8", "#22c55e"])
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
with c4:
    fig4 = px.bar(agg, x="Path", y="Total latency (s)", color="Path",
                  color_discrete_sequence=["#94a3b8", "#3b82f6"])
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()


st.subheader("Quality vs. latency, per prompt")

scatter_rows = []
for r in results:
    scatter_rows.append({"path": "Baseline", "tier": r["baseline"]["model_tier"],
                          "latency_s": r["baseline"]["latency_s"],
                          "quality": r["baseline"]["quality_score"],
                          "prompt": r["prompt"][:50]})
    scatter_rows.append({"path": "Router", "tier": r["router"]["model_tier"],
                          "latency_s": r["router"]["latency_s"],
                          "quality": r["router"]["quality_score"],
                          "prompt": r["prompt"][:50]})
scatter_df = pd.DataFrame(scatter_rows)

fig5 = px.scatter(scatter_df, x="latency_s", y="quality", color="path", symbol="tier",
                   hover_data=["prompt"], opacity=0.75,
                   labels={"latency_s": "Latency (s)", "quality": "Quality score (1-5)"})
fig5.update_traces(marker=dict(size=11))
st.plotly_chart(fig5, use_container_width=True)

st.divider()

st.subheader("Generation speed by model tier (tokens/sec)")
speed_rows = []
for r in results:
    speed_rows.append({"tier": r["router"]["model_tier"], "tokens_per_sec": r["router"].get("tokens_per_sec", 0)})
speed_df = pd.DataFrame(speed_rows)
if not speed_df.empty:
    speed_avg = speed_df.groupby("tier", as_index=False)["tokens_per_sec"].mean()
    fig6 = px.bar(speed_avg, x="tier", y="tokens_per_sec", color="tier",
                  category_orders={"tier": ["tiny", "small", "mid", "large"]},
                  labels={"tokens_per_sec": "Avg tokens/sec"})
    fig6.update_layout(showlegend=False)
    st.plotly_chart(fig6, use_container_width=True)

st.divider()

st.subheader("Per-prompt comparison")

rows = []
for r in results:
    rows.append({
        "Prompt": r["prompt"][:70] + ("..." if len(r["prompt"]) > 70 else ""),
        "Expected type": r["expected_label"],
        "Routed to": f"{r['router']['predicted_label']} → {r['router']['model_tier']}",
        "Correct route?": "✅" if r["router"]["classifier_correct"] else "❌",
        "Baseline cost ($)": round(r["baseline"]["cost_usd"], 6),
        "Router cost ($)": round(r["router"]["cost_usd"], 6),
        "Baseline quality": r["baseline"]["quality_score"],
        "Router quality": r["router"]["quality_score"],
        "Baseline latency (s)": round(r["baseline"]["latency_s"], 2),
        "Router latency (s)": round(r["router"]["latency_s"], 2),
    })

table_df = pd.DataFrame(rows)
st.dataframe(table_df, use_container_width=True, height=500)

st.divider()


st.subheader("Inspect a single response")
idx = st.selectbox(
    "Pick a prompt",
    options=range(len(results)),
    format_func=lambda i: results[i]["prompt"][:80],
)
r = results[idx]

d1, d2 = st.columns(2)
with d1:
    st.markdown(f"**Baseline ({r['baseline']['model_id']})**")
    st.caption(f"Cost: ${r['baseline']['cost_usd']:.6f} | Latency: {r['baseline']['latency_s']:.2f}s | "
               f"{r['baseline'].get('tokens_per_sec', 0):.1f} tok/s | Quality: {r['baseline']['quality_score']}/5")
    st.write(r["baseline"]["text"])
    if r["baseline"].get("text_raw") and r["baseline"]["text_raw"] != r["baseline"]["text"]:
        with st.expander("Show raw output (incl. model's thinking)"):
            st.text(r["baseline"]["text_raw"])
with d2:
    st.markdown(f"**Router → {r['router']['model_id']} ({r['router']['model_tier']} tier)**")
    st.caption(f"Cost: ${r['router']['cost_usd']:.6f} | Latency: {r['router']['latency_s']:.2f}s | "
               f"{r['router'].get('tokens_per_sec', 0):.1f} tok/s | Quality: {r['router']['quality_score']}/5")
    st.write(r["router"]["text"])
    if r["router"].get("text_raw") and r["router"]["text_raw"] != r["router"]["text"]:
        with st.expander("Show raw output (incl. model's thinking)"):
            st.text(r["router"]["text_raw"])

st.caption(f"Benchmark generated at {summary['generated_at']}")
