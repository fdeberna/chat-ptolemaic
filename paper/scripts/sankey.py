import plotly.graph_objects as go

models = {
    "Base": {
        "P_pre": 0.309,
        "P_geo_pre": 0.243,
        "P_helio_pre": 0.127,
        "P_helio_modern": 0.687,
    },
    "QLoRA-1000": {
        "P_pre": 0.654,
        "P_geo_pre": 0.235,
        "P_helio_pre": 0.022,
        "P_helio_modern": 0.667,
    },
}

def sankey_for_model(name, vals):
    p_pre = vals["P_pre"]
    p_mod = 1 - p_pre

    # Assumption: modern geocentric mass is whatever remains needed
    # to match overall geo rate; set to 0 here unless you have it explicitly.
    p_geo_pre = vals["P_geo_pre"]
    p_helio_pre = vals["P_helio_pre"]
    p_helio_mod = vals["P_helio_modern"]

    flows = {
        ("Premodern", "Geocentric"): p_pre * p_geo_pre,
        ("Premodern", "Heliocentric"): p_pre * p_helio_pre,
        ("Premodern", "Other / ambiguous"): p_pre * (1 - p_geo_pre - p_helio_pre),
        ("Modern", "Heliocentric"): p_mod * p_helio_mod,
        ("Modern", "Other / ambiguous"): p_mod * (1 - p_helio_mod),
    }

    labels = ["Premodern", "Modern", "Geocentric", "Heliocentric", "Other / ambiguous"]
    idx = {label: i for i, label in enumerate(labels)}

    source = [idx[a] for a, b in flows]
    target = [idx[b] for a, b in flows]
    value = [v * 100 for v in flows.values()]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            label=labels,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=[f"{a} → {b}: {v*100:.1f}%" for (a, b), v in flows.items()],
        )
    )])

    fig.update_layout(
        title_text=f"Frame-to-stance probability flow: {name}",
        font_size=14,
        width=700,
        height=450,
    )

    return fig

for name, vals in models.items():
    fig = sankey_for_model(name, vals)
    fig.write_image(f"sankey_{name.lower().replace('-', '_')}.pdf")
    fig.write_html(f"sankey_{name.lower().replace('-', '_')}.html")
    # fig.show()