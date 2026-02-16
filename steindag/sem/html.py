"""HTML-related utilities for SEM visualization/export."""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import torch
from torch import Tensor

from steindag.variable.functional import FunctionalVariable
from steindag.variable.linear import LinearVariable


class HTMLMixin:
    """Mixin with HTML rendering/export functionality for SEM artifacts."""

    def export_html(
        self,
        output_path: str | Path,
        *,
        observed_ranges: dict[str, tuple[float, float] | tuple[float, float, int]],
        baseline_observed: dict[str, float] | None = None,
        outcome_name: str | None = None,
        num_posterior_samples: int = 400,
        max_precomputed_states: int = 1200,
        title: str = "SCM Explorer",
    ) -> Path:
        """Export an interactive HTML explorer with sliders and SEM context."""
        output = Path(output_path)
        if not observed_ranges:
            raise ValueError("`observed_ranges` must contain at least one variable.")
        if num_posterior_samples <= 0:
            raise ValueError("`num_posterior_samples` must be > 0.")
        if max_precomputed_states <= 0:
            raise ValueError("`max_precomputed_states` must be > 0.")

        slider_names = list(observed_ranges.keys())
        unknown = [name for name in slider_names if name not in self._variables]
        if unknown:
            raise ValueError(
                "Unknown slider variable(s): " + ", ".join(sorted(unknown))
            )

        baseline = dict(baseline_observed or {})
        unknown_baseline = [name for name in baseline if name not in self._variables]
        if unknown_baseline:
            raise ValueError(
                "Unknown baseline variable(s): " + ", ".join(sorted(unknown_baseline))
            )

        slider_values: list[list[float]] = []
        slider_steps: list[int] = []
        slider_initial_idx: list[int] = []

        for name in slider_names:
            spec = observed_ranges[name]
            if len(spec) == 2:
                low, high = spec
                steps = 9
            else:
                low, high, steps = spec
            if not high > low:
                raise ValueError(f"Range for '{name}' must satisfy max > min.")
            if steps < 2:
                raise ValueError(f"Range steps for '{name}' must be >= 2.")

            grid = torch.linspace(float(low), float(high), int(steps)).tolist()
            slider_values.append([float(v) for v in grid])
            slider_steps.append(int(steps))

            target = float(baseline.get(name, 0.5 * (float(low) + float(high))))
            baseline[name] = target
            closest_idx = min(
                range(len(grid)), key=lambda i: abs(float(grid[i]) - target)
            )
            slider_initial_idx.append(closest_idx)

        total_states = 1
        for steps in slider_steps:
            total_states *= steps
        if total_states > max_precomputed_states:
            raise ValueError(
                "Cross-product grid too large: "
                f"{total_states} states exceeds max_precomputed_states={max_precomputed_states}."
            )

        observed_names = set(baseline.keys()) | set(slider_names)
        variable_names = [
            name for name in self._variables.keys() if name not in observed_names
        ]
        if not variable_names:
            raise ValueError(
                "No latent/unobserved variables left to plot. "
                "Provide fewer observed variables in `observed_ranges`/`baseline_observed`."
            )

        causal_effect_names: list[str] = []
        causal_bias_names: list[str] = []
        causal_treatments: list[str] = []
        if outcome_name is not None:
            if outcome_name not in self._variables:
                raise ValueError(f"Unknown outcome variable '{outcome_name}'.")
            if outcome_name in observed_names:
                raise ValueError(
                    "`outcome_name` cannot be included in observed variables used by sliders/baseline."
                )

            candidate_observed = {
                name: torch.tensor([0.0], dtype=torch.float32) for name in observed_names
            }
            for treatment_name in sorted(observed_names):
                self._validate_causal_query(
                    candidate_observed,
                    treatment_name=treatment_name,
                    outcome_name=outcome_name,
                )
                causal_treatments.append(treatment_name)
                causal_effect_names.append(
                    f"causal_effect({treatment_name}->{outcome_name})"
                )
                causal_bias_names.append(f"causal_bias({treatment_name}->{outcome_name})")

        asset_dir = output.parent / f"{output.stem}_assets"
        asset_dir.mkdir(parents=True, exist_ok=True)

        dag_img = self.draw(
            output_path=asset_dir / "dag.png",
            observed_names=sorted(observed_names),
            title=None,
        )

        flow_animation_assets: list[dict[str, str]] = []
        if outcome_name is not None:
            for treatment_name in causal_treatments:
                flow_path = asset_dir / f"flow_{treatment_name}_to_{outcome_name}.gif"
                try:
                    out_flow = self.animate(
                        output_path=flow_path,
                        observed_names=sorted(observed_names),
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                        fps=10,
                        frames_per_flow=20,
                        title=None,
                    )
                except ValueError:
                    continue
                flow_animation_assets.append(
                    {
                        "label": f"{treatment_name} → {outcome_name}",
                        "src": out_flow.relative_to(output.parent).as_posix(),
                    }
                )

        plot_names = [*variable_names, *causal_effect_names, *causal_bias_names]
        states: list[dict[str, object]] = []
        x_ranges: dict[str, list[float]] = {
            name: [float("inf"), float("-inf")] for name in plot_names
        }

        for slider_idx_tuple in product(*[range(s) for s in slider_steps]):
            observed_vals = dict(baseline)
            for i, name in enumerate(slider_names):
                observed_vals[name] = slider_values[i][slider_idx_tuple[i]]

            observed = {
                name: torch.tensor([value], dtype=torch.float32)
                for name, value in observed_vals.items()
            }
            posterior_samples = self.posterior_predictive_samples(
                observed=observed,
                num_samples=num_posterior_samples,
            )

            by_variable: dict[str, list[float]] = {}
            for name in variable_names:
                series = posterior_samples[name].squeeze(0).detach().cpu()
                values = [float(v) for v in series.tolist()]
                by_variable[name] = values
                if values:
                    x_ranges[name][0] = min(x_ranges[name][0], min(values))
                    x_ranges[name][1] = max(x_ranges[name][1], max(values))

            if outcome_name is not None:
                for treatment_name in causal_treatments:
                    effect_samples = self._compute_causal_effect_samples(
                        observed=observed,
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                        num_samples=num_posterior_samples,
                    )
                    bias_samples = self._compute_causal_bias_samples(
                        observed=observed,
                        treatment_name=treatment_name,
                        outcome_name=outcome_name,
                        num_samples=num_posterior_samples,
                    )
                    effect_name = f"causal_effect({treatment_name}->{outcome_name})"
                    bias_name = f"causal_bias({treatment_name}->{outcome_name})"
                    effect_values = [
                        float(v) for v in effect_samples.squeeze(0).detach().cpu().tolist()
                    ]
                    bias_values = [
                        float(v) for v in bias_samples.squeeze(0).detach().cpu().tolist()
                    ]
                    by_variable[effect_name] = effect_values
                    by_variable[bias_name] = bias_values

                    if effect_values:
                        x_ranges[effect_name][0] = min(
                            x_ranges[effect_name][0], min(effect_values)
                        )
                        x_ranges[effect_name][1] = max(
                            x_ranges[effect_name][1], max(effect_values)
                        )
                    if bias_values:
                        x_ranges[bias_name][0] = min(x_ranges[bias_name][0], min(bias_values))
                        x_ranges[bias_name][1] = max(x_ranges[bias_name][1], max(bias_values))

            states.append(
                {
                    "slider_indices": list(slider_idx_tuple),
                    "samples": by_variable,
                }
            )

        for name in plot_names:
            lo, hi = x_ranges[name]
            if not (lo < hi):
                lo -= 1.0
                hi += 1.0
            pad = 0.05 * (hi - lo)
            x_ranges[name] = [lo - pad, hi + pad]

        payload = {
            "title": title,
            "variable_names": plot_names,
            "plot_groups": {
                "variables": variable_names,
                "causal_effects": causal_effect_names,
                "causal_biases": causal_bias_names,
                "flow_animations": [item["label"] for item in flow_animation_assets],
            },
            "scm": {
                "equations": self._build_scm_equations_payload(),
            },
            "dag": {
                "nodes": [
                    {
                        "name": name,
                        "x": float(self._compute_plot_positions()[name][0]),
                        "y": float(self._compute_plot_positions()[name][1]),
                        "observed": name in observed_names,
                        "outcome": (outcome_name is not None and name == outcome_name),
                    }
                    for name in self._variables.keys()
                ],
                "edges": [
                    {"source": parent_name, "target": child_name}
                    for child_name, variable in self._variables.items()
                    for parent_name in variable.parent_names
                ],
            },
            "slider_names": slider_names,
            "slider_values": slider_values,
            "slider_initial_idx": slider_initial_idx,
            "states": states,
            "x_ranges": x_ranges,
            "assets": {
                "dag_image": dag_img.relative_to(output.parent).as_posix(),
                "flow_animations": flow_animation_assets,
            },
        }

        html = self.display_html(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        return output

    def _build_scm_equations_payload(self) -> list[dict[str, object]]:
        equations: list[dict[str, object]] = []

        for name, variable in self._variables.items():
            parameters: list[dict[str, object]] = []

            sigma_symbol = rf"\sigma_{{{name}}}"
            sigma_value = 1.0 if variable.sigma is None else float(variable.sigma)
            parameters.append({"symbol": sigma_symbol, "value": sigma_value})

            if isinstance(variable, LinearVariable):
                terms: list[str] = []

                intercept = float(getattr(variable, "_intercept", 0.0))
                alpha_symbol = rf"\alpha_{{{name}}}"
                parameters.append({"symbol": alpha_symbol, "value": intercept})
                terms.append(alpha_symbol)

                coefs = dict(getattr(variable, "_coefs", {}))
                for parent_name in variable.parent_names:
                    beta_symbol = rf"\beta_{{{name},{parent_name}}}"
                    beta_value = float(coefs.get(parent_name, 0.0))
                    parameters.append({"symbol": beta_symbol, "value": beta_value})
                    terms.append(f"{beta_symbol}\\,{parent_name}")

                rhs_main = " + ".join(terms) if terms else "0"
            elif isinstance(variable, FunctionalVariable):
                parent_expr = ", ".join(variable.parent_names)
                rhs_main = (
                    rf"f_{{{name}}}\left({parent_expr}\right)"
                    if parent_expr
                    else rf"f_{{{name}}}"
                )

                coefs = getattr(variable, "_coefs", None)
                if coefs:
                    for parent_name, value in coefs.items():
                        theta_symbol = rf"\theta_{{{name},{parent_name}}}"
                        parameters.append(
                            {"symbol": theta_symbol, "value": float(value)}
                        )

                intercept = getattr(variable, "_intercept", None)
                if intercept is not None:
                    gamma_symbol = rf"\gamma_{{{name}}}"
                    parameters.append({"symbol": gamma_symbol, "value": float(intercept)})
            else:
                parent_expr = ", ".join(variable.parent_names)
                rhs_main = (
                    rf"f_{{{name}}}\left({parent_expr}\right)"
                    if parent_expr
                    else rf"f_{{{name}}}"
                )

            equation_latex = rf"{name} = {rhs_main} + {sigma_symbol}\,U_{{{name}}}"

            equations.append(
                {
                    "name": name,
                    "equation_latex": equation_latex,
                    "parameters": parameters,
                }
            )

        return equations

    @staticmethod
    def display_html(payload: dict[str, object]) -> str:
        """Render the full interactive HTML page for a given payload."""
        data_json = json.dumps(payload)
        return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>SteinDAG Explorer</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['\\(', '\\)'], ['$', '$']],
        displayMath: [['\\[', '\\]']],
        processEscapes: true,
      }},
      svg: {{ fontCache: 'global' }},
    }};
  </script>
  <script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js\"></script>
  <style>
    :root {{
      --bg: #0f1117;
      --panel: #171b25;
      --card: #1f2533;
      --text: #f1f5ff;
      --muted: #9aa6bf;
      --accent: #66b3ff;
    }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
      background: linear-gradient(180deg, #0c0f15, #111623);
      color: var(--text);
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 20px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 16px;
      align-items: start;
    }}
    .dag-panel {{
      background: var(--panel);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 10px;
      margin-bottom: 12px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.18);
    }}
    .dag-title {{
      font-size: 14px;
      color: var(--muted);
      margin: 0 0 4px 2px;
    }}
    .top-schedules {{
      display: flex;
      gap: 8px;
      margin: 2px 0 8px 0;
      flex-wrap: wrap;
    }}
    .top-schedule-btn {{
      background: #121723;
      color: var(--text);
      border: 1px solid #2a3142;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 12px;
      cursor: pointer;
    }}
    .top-schedule-btn.active {{
      background: #234163;
      border-color: #4479b3;
    }}
    .dag-plot {{ width: 100%; height: 240px; }}
    .dag-image {{ width: 100%; height: 240px; object-fit: contain; background: #171b25; border: 1px solid #2a3142; border-radius: 8px; }}
    .top-view {{
      margin-top: 8px;
    }}
    .scm-top-title {{
      font-size: 13px;
      color: var(--muted);
      margin: 0 0 6px 0;
    }}
    .main {{
      background: var(--panel);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }}
    .schedules {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }}
    .schedule-btn {{
      background: #121723;
      color: var(--text);
      border: 1px solid #2a3142;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 12px;
      cursor: pointer;
    }}
    .schedule-btn.active {{
      background: #234163;
      border-color: #4479b3;
    }}
    .scm-card {{
      background: var(--card);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 14px;
      display: grid;
      gap: 12px;
    }}
    .scm-eq {{
      border: 1px solid #2a3142;
      border-radius: 10px;
      padding: 10px;
      background: #131927;
    }}
    .scm-eq-grid {{
      display: grid;
      grid-template-columns: max-content 22px minmax(0, 1fr);
      column-gap: 6px;
      row-gap: 2px;
      align-items: baseline;
    }}
    .scm-eq-row {{
      display: contents;
    }}
    .scm-eq-equals {{
      text-align: center;
      color: #dbe4f8;
      font-weight: 600;
    }}
    .scm-param-list {{
      margin: 8px 0 0 0;
      padding-left: 18px;
      color: #cdd6ea;
      font-size: 12px;
    }}
    .sidebar {{
      position: sticky;
      top: 14px;
      background: var(--panel);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 8px 20px rgba(0,0,0,0.25);
    }}
    .sliders {{ display: grid; gap: 10px; }}
    .slider-row {{
      background: #121723;
      border: 1px solid #2a3142;
      border-radius: 10px;
      padding: 8px;
    }}
    .slider-row label {{
      display: flex;
      justify-content: space-between;
      font-weight: 600;
      margin-bottom: 4px;
      font-size: 13px;
    }}
    input[type=range] {{ width: 100%; accent-color: var(--accent); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 8px;
    }}
    .plot {{ width: 100%; height: 280px; }}
    .flow-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .flow-card {{
      background: var(--card);
      border: 1px solid #2a3142;
      border-radius: 12px;
      padding: 10px;
    }}
    .flow-card h4 {{ margin: 0 0 8px 0; font-size: 13px; color: #cfd7ec; }}
    .flow-card img {{ width: 100%; border-radius: 8px; border: 1px solid #2a3142; }}
    .sidebar h2 {{ margin: 0 0 8px 0; font-size: 16px; }}
    .sidebar .hint {{ color: var(--muted); font-size: 12px; margin-bottom: 10px; }}
    @media (max-width: 1020px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; order: -1; }}
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"dag-panel\">
      <div class=\"dag-title\">DAG structure</div>
      <div id=\"top-schedules\" class=\"top-schedules\"></div>
      <div id=\"dag-view\" class=\"top-view\">
        <img id=\"dag-image\" class=\"dag-image\" alt=\"DAG\" />
      </div>
      <div id=\"scm-view\" class=\"top-view\" style=\"display:none\">
        <h3 class=\"scm-top-title\">Structural causal model</h3>
        <div id=\"scm\" class=\"scm-card\"></div>
      </div>
    </div>
    <div class=\"layout\">
      <div class=\"main\">
        <div id=\"schedules\" class=\"schedules\"></div>
        <div id=\"plots\" class=\"grid\"></div>
      </div>
      <aside class=\"sidebar\">
        <h2>Observed values</h2>
        <div class=\"hint\">Move sliders to switch between precomputed posterior states.</div>
        <div id=\"sliders\" class=\"sliders\"></div>
      </aside>
    </div>
  </div>

  <script>
    const DATA = {data_json};
    const sliderIndices = [...DATA.slider_initial_idx];
    const GROUP_LABELS = {{
      variables: 'Variable distributions',
      causal_effects: 'Causal effects',
      causal_biases: 'Causal biases',
      flow_animations: 'Flow animations',
    }};
    const TOP_GROUP_LABELS = {{
      dag: 'DAG structure',
      scm: 'Structural causal model (LaTeX)',
    }};
    const groupOrder = ['variables', 'causal_effects', 'causal_biases', 'flow_animations'];
    const availableGroups = groupOrder.filter((k) => (DATA.plot_groups[k] || []).length > 0);
    let activeGroup = availableGroups[0] || 'variables';
    let activeTopGroup = 'dag';
    const sliderEls = [];
    const valueEls = [];
    let mathJaxRetryAttempts = 0;

    function tryTypeset(root) {{
      if (!root) return false;
      if (window.MathJax && window.MathJax.typesetPromise) {{
        root.dataset.needsTypeset = '0';
        window.MathJax.typesetPromise([root]).catch(() => {{
          root.dataset.needsTypeset = '1';
        }});
        return true;
      }}
      root.dataset.needsTypeset = '1';
      return false;
    }}

    function scheduleMathJaxRetry() {{
      const root = document.getElementById('scm');
      if (!root || root.dataset.needsTypeset !== '1') return;
      if (tryTypeset(root)) return;
      if (mathJaxRetryAttempts >= 30) return;
      mathJaxRetryAttempts += 1;
      setTimeout(scheduleMathJaxRetry, 250);
    }}

    function setTopView(groupKey) {{
      activeTopGroup = groupKey;
      const showDag = groupKey === 'dag';
      document.getElementById('dag-view').style.display = showDag ? 'block' : 'none';
      document.getElementById('scm-view').style.display = showDag ? 'none' : 'block';
      renderTopSchedules();
      if (!showDag) {{
        renderScm();
      }}
    }}

    function renderTopSchedules() {{
      const root = document.getElementById('top-schedules');
      root.innerHTML = '';
      ['dag', 'scm'].forEach((groupKey) => {{
        const btn = document.createElement('button');
        btn.className = `top-schedule-btn ${{groupKey === activeTopGroup ? 'active' : ''}}`;
        btn.textContent = TOP_GROUP_LABELS[groupKey] || groupKey;
        btn.onclick = () => setTopView(groupKey);
        root.appendChild(btn);
      }});
    }}

    function stateOffset(indices) {{
      let offset = 0;
      let stride = 1;
      for (let i = DATA.slider_values.length - 1; i >= 0; i--) {{
        offset += indices[i] * stride;
        stride *= DATA.slider_values[i].length;
      }}
      return offset;
    }}

    function currentState() {{
      return DATA.states[stateOffset(sliderIndices)];
    }}

    function renderSliders() {{
      const root = document.getElementById('sliders');
      root.innerHTML = '';
      DATA.slider_names.forEach((name, i) => {{
        const row = document.createElement('div');
        row.className = 'slider-row';
        const label = document.createElement('label');
        const left = document.createElement('span');
        left.textContent = name;
        const right = document.createElement('span');
        right.textContent = DATA.slider_values[i][sliderIndices[i]].toFixed(3);
        label.appendChild(left);
        label.appendChild(right);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = 0;
        slider.max = DATA.slider_values[i].length - 1;
        slider.step = 1;
        slider.value = sliderIndices[i];
        slider.oninput = (e) => {{
          sliderIndices[i] = Number(e.target.value);
          right.textContent = DATA.slider_values[i][sliderIndices[i]].toFixed(3);
          renderPlots();
        }};

        row.appendChild(label);
        row.appendChild(slider);
        root.appendChild(row);
        sliderEls.push(slider);
        valueEls.push(right);
      }});
    }}

    function currentNames() {{
      return DATA.plot_groups[activeGroup] || [];
    }}

    function renderSchedules() {{
      const root = document.getElementById('schedules');
      root.innerHTML = '';
      availableGroups.forEach((groupKey) => {{
        const btn = document.createElement('button');
        btn.className = `schedule-btn ${{groupKey === activeGroup ? 'active' : ''}}`;
        btn.textContent = GROUP_LABELS[groupKey] || groupKey;
        btn.onclick = () => {{
          activeGroup = groupKey;
          renderSchedules();
          renderPlotContainers();
          renderPlots();
        }};
        root.appendChild(btn);
      }});
    }}

    function renderPlotContainers() {{
      const root = document.getElementById('plots');
      root.innerHTML = '';
      if (activeGroup === 'flow_animations') {{
        const wrap = document.createElement('div');
        wrap.className = 'flow-grid';
        (DATA.assets?.flow_animations || []).forEach((item) => {{
          const card = document.createElement('div');
          card.className = 'flow-card';
          const h = document.createElement('h4');
          h.textContent = item.label;
          const img = document.createElement('img');
          img.src = item.src;
          img.alt = item.label;
          card.appendChild(h);
          card.appendChild(img);
          wrap.appendChild(card);
        }});
        root.appendChild(wrap);
        return;
      }}
      currentNames().forEach((name, i) => {{
        const card = document.createElement('div');
        card.className = 'card';
        const div = document.createElement('div');
        div.className = 'plot';
        div.id = `plot-${{i}}`;
        card.appendChild(div);
        root.appendChild(card);
      }});
    }}

    function renderPlots() {{
      if (activeGroup === 'flow_animations') return;
      const st = currentState();
      currentNames().forEach((name, i) => {{
        const samples = st.samples[name];
        const nbins = Math.max(15, Math.round(Math.sqrt(samples.length || 1)));
        const trace = {{
          x: samples,
          type: 'histogram',
          histnorm: 'probability density',
          marker: {{ color: '#66b3ff' }},
          opacity: 0.9,
          nbinsx: nbins,
        }};
        const mean = samples.reduce((a, b) => a + b, 0) / Math.max(samples.length, 1);
        const layout = {{
          title: {{ text: `${{name}} &nbsp; <span style=\"font-size:12px;color:#9aa6bf\">mean=${{mean.toFixed(3)}}</span>` }},
          margin: {{ l: 40, r: 10, t: 40, b: 36 }},
          paper_bgcolor: '#1f2533',
          plot_bgcolor: '#1f2533',
          font: {{ color: '#f1f5ff' }},
          xaxis: {{ range: DATA.x_ranges[name], gridcolor: '#2f394d' }},
          yaxis: {{ gridcolor: '#2f394d' }},
        }};
        Plotly.react(`plot-${{i}}`, [trace], layout, {{ displayModeBar: false, responsive: true }});
      }});
    }}

    function renderScm() {{
      const root = document.getElementById('scm');
      if (!root) return;
      root.innerHTML = '';
      mathJaxRetryAttempts = 0;
      const equations = DATA.scm?.equations || [];
      if (equations.length) {{
        const eqBlock = document.createElement('div');
        eqBlock.className = 'scm-eq scm-eq-grid';
        equations.forEach((eq) => {{
          const parts = eq.equation_latex.split(' = ');
          if (parts.length >= 2) {{
            const lhs = parts[0];
            const rhs = parts.slice(1).join(' = ');
            const row = document.createElement('div');
            row.className = 'scm-eq-row';

            const lhsEl = document.createElement('div');
            lhsEl.textContent = '\\(' + lhs + '\\)';

            const eqEl = document.createElement('div');
            eqEl.className = 'scm-eq-equals';
            eqEl.textContent = '=';

            const rhsEl = document.createElement('div');
            rhsEl.textContent = '\\(' + rhs + '\\)';

            row.appendChild(lhsEl);
            row.appendChild(eqEl);
            row.appendChild(rhsEl);
            eqBlock.appendChild(row);
          }} else {{
            const row = document.createElement('div');
            row.textContent = '\\(' + eq.equation_latex + '\\)';
            eqBlock.appendChild(row);
          }}
        }});
        root.appendChild(eqBlock);
      }}

      const allParams = equations.flatMap((eq) => eq.parameters || []);
      const noiseTerms = equations.map((eq) => `U_{{${{eq.name}}}}`);
      if (allParams.length || noiseTerms.length) {{
        const paramBlock = document.createElement('div');
        paramBlock.className = 'scm-eq';
        const ul = document.createElement('ul');
        ul.className = 'scm-param-list';

        allParams.forEach((p) => {{
          const li = document.createElement('li');
          li.textContent = '\\(' + p.symbol + ' = ' + Number(p.value).toFixed(4) + '\\)';
          ul.appendChild(li);
        }});

        if (noiseTerms.length) {{
          const li = document.createElement('li');
          li.textContent = '\(' + noiseTerms.join(', ') + ' \\overset{{\\mathrm{{iid}}}}{{\\sim}} \\mathcal{{N}}(0, 1)' + '\)';
          ul.appendChild(li);
        }}

        paramBlock.appendChild(ul);
        root.appendChild(paramBlock);
      }}

      if (!root.children.length) {{
        const empty = document.createElement('div');
        empty.style.color = '#cdd6ea';
        empty.style.fontSize = '12px';
        empty.textContent = 'No structural equations available.';
        root.appendChild(empty);
      }}

      if (!tryTypeset(root)) {{
        scheduleMathJaxRetry();
      }}
    }}

    function renderDag() {{
      const img = document.getElementById('dag-image');
      if (img && DATA.assets?.dag_image) {{
        img.src = DATA.assets.dag_image;
      }}
    }}

    renderTopSchedules();
    setTopView('dag');
    renderDag();
    renderSliders();
    renderSchedules();
    renderPlotContainers();
    renderPlots();

    window.addEventListener('load', () => {{
      const root = document.getElementById('scm');
      if (root && root.dataset.needsTypeset === '1') {{
        tryTypeset(root);
      }}
    }});
  </script>
</body>
</html>
"""
