"""Plotting and flow-animation mixin for Structural Causal Models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, TypedDict, cast
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


class _TraversalProtocol(Protocol):
    def _get_descendants(self, variable_name: str) -> set[str]: ...

    def _get_ancestors(self, variable_name: str) -> set[str]: ...


class _EdgeGeometry(TypedDict):
    start: tuple[float, float]
    end: tuple[float, float]
    rad: float


class FlowStage(TypedDict):
    name: str
    label: str
    color: str
    linestyle: str
    pulse_nodes: list[str]
    paths: list[list[str]]


class PlottingMixin:
    """Mixin with DAG plotting and staged flow animation utilities."""

    _variables: dict

    _FLOW_COLORS = {
        "Causal path": "#2ECC71",
        "Confounder bias": "#FF4D4F",
        "Mediator bias": "#F5A623",
        "Selection bias": "#9B59B6",
    }

    def draw(
        self,
        output_path: str | Path,
        *,
        observed_names: Iterable[str] = (),
        figsize: tuple[float, float] = (9.0, 5.0),
        title: str | None = None,
        dpi: int = 180,
    ) -> Path:
        """Draw a static DAG view with observed/unobserved node styling.

        This method provides a simplified counterpart to :meth:`animate`,
        without treatment/outcome flow annotations.
        """
        observed = set(observed_names)
        self._validate_observed_names(observed)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        self._draw_flow_base(
            ax=ax,
            observed=observed,
            treatment_name=None,
            outcome_name=None,
        )

        if title:
            ax.set_title(title, fontsize=11, color="#1D1D1F")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output

    def animate_flow_gif(
        self,
        output_path: str | Path,
        *,
        observed_names: Iterable[str],
        treatment_name: str,
        outcome_name: str,
        fps: int = 15,
        frames_per_flow: int = 36,
        figsize: tuple[float, float] = (9.0, 5.0),
        title: str | None = None,
    ) -> Path:
        """Backward-compatible alias for :meth:`animate`."""
        return self.animate(
            output_path=output_path,
            observed_names=observed_names,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            fps=fps,
            frames_per_flow=frames_per_flow,
            figsize=figsize,
            title=title,
        )

    def path_flows(
        self,
        *,
        treatment_name: str,
        outcome_name: str,
        observed_names: Iterable[str],
    ) -> dict[str, list[list[str]]]:
        """Return categorized open/blocked path flows for plotting."""
        observed = set(observed_names)
        self._validate_observed_names(observed)
        self._validate_treatment_outcome(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=observed,
        )
        return self._collect_flow_paths(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=observed,
        )

    def plot_dag(
        self,
        *,
        observed_names: Iterable[str],
        treatment_name: str | None = None,
        outcome_name: str | None = None,
        figsize: tuple[float, float] = (9.0, 5.0),
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Render a DAG and return the matplotlib figure/axes pair."""
        observed = set(observed_names)
        self._validate_observed_names(observed)

        if (treatment_name is None) != (outcome_name is None):
            raise ValueError(
                "`treatment_name` and `outcome_name` must be both provided or both omitted."
            )
        if treatment_name is not None and outcome_name is not None:
            self._validate_treatment_outcome(
                treatment_name=treatment_name,
                outcome_name=outcome_name,
                observed=observed,
            )

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        _, edge_geometry = self._draw_flow_base(
            ax=ax,
            observed=observed,
            treatment_name=None,
            outcome_name=None,
        )

        if treatment_name is not None and outcome_name is not None:
            categories = self._compute_path_edge_categories(
                treatment_name=treatment_name,
                outcome_name=outcome_name,
                conditioned_on=observed,
            )
            category_styles = {
                "open_causal": ("#2ECC71", "solid"),
                "open_noncausal": ("#FF4D4F", "solid"),
                "blocked_causal": ("#F5A623", "dashed"),
            }
            for edge, edge_categories in categories.items():
                geom = edge_geometry.get(edge)
                if geom is None:
                    continue
                edge_name = f"{edge[0]}->{edge[1]}"
                for category in sorted(edge_categories):
                    color, linestyle = category_styles.get(
                        category,
                        ("#8E44AD", "solid"),
                    )
                    arrow = FancyArrowPatch(
                        geom["start"],
                        geom["end"],
                        arrowstyle="->",
                        mutation_scale=10,
                        linewidth=2.0,
                        color=color,
                        linestyle=linestyle,
                        connectionstyle=f"arc3,rad={geom['rad']}",
                        shrinkA=0,
                        shrinkB=0,
                        clip_on=False,
                        alpha=0.95,
                        zorder=3,
                    )
                    arrow.set_gid(f"edge:{edge_name}:{category}")
                    ax.add_patch(arrow)

            for patch in ax.patches:
                gid = getattr(patch, "get_gid", lambda: None)()
                if gid is None or not str(gid).startswith("node:"):
                    continue
                node_name = str(gid).split(":", 1)[1]
                if node_name == treatment_name:
                    patch.set_facecolor("#2E5BFF")
                elif node_name == outcome_name:
                    patch.set_facecolor("#8E44AD")

        if title:
            ax.set_title(title, fontsize=11, color="#1D1D1F")

        return fig, ax

    def animate(
        self,
        output_path: str | Path,
        *,
        observed_names: Iterable[str],
        treatment_name: str,
        outcome_name: str,
        fps: int = 15,
        frames_per_flow: int = 36,
        figsize: tuple[float, float] = (9.0, 5.0),
        title: str | None = None,
    ) -> Path:
        observed = set(observed_names)
        self._validate_observed_names(observed)
        self._validate_treatment_outcome(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=observed,
        )
        stages = self.flow_animation_stages(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed_names=observed,
        )
        if not stages:
            raise ValueError("No valid treatment->outcome flow paths to animate.")

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        positions, edge_geometry = self._draw_flow_base(
            ax=ax,
            observed=observed,
            treatment_name=treatment_name,
            outcome_name=outcome_name,
        )

        pulse_nodes = ax.scatter([], [], s=[], c=[], alpha=0.88, zorder=6)
        pulse_halo = ax.scatter([], [], s=[], c=[], alpha=0.34, zorder=5)
        active_arrows: list[FancyArrowPatch] = []

        frames_per_stage = max(frames_per_flow, 2)
        hold_frames = max(int(round(0.5 * fps)), 1)
        segment_frames = frames_per_stage + hold_frames

        playback_items: list[tuple[FlowStage, list[str]]] = []
        for stage in stages:
            playback_items.extend((stage, path) for path in stage["paths"])
        if not playback_items:
            raise ValueError("No path items available for animation playback.")

        if title:
            ax.set_title(title, fontsize=11, color="#1D1D1F")

        def update(frame_idx: int):
            nonlocal active_arrows
            for arrow in active_arrows:
                arrow.remove()
            active_arrows = []

            item_idx = (frame_idx // segment_frames) % len(playback_items)
            local = frame_idx % segment_frames
            progress = (
                1.0
                if local >= frames_per_stage
                else local / max(frames_per_stage - 1, 1)
            )
            stage, active_path = playback_items[item_idx]

            color = self._FLOW_COLORS.get(stage["label"], stage["color"])

            path_edges = [
                self._path_segment_to_directed_edge(active_path[i], active_path[i + 1])
                for i in range(len(active_path) - 1)
            ]
            scaled = progress * len(path_edges)
            full_edges = int(min(np.floor(scaled + 1e-10), len(path_edges)))
            partial_alpha = float(np.clip(scaled - full_edges, 0.0, 1.0))
            for i, edge in enumerate(path_edges):
                geom = edge_geometry.get(edge)
                if geom is None:
                    continue
                alpha = (
                    1.0
                    if i < full_edges
                    else (partial_alpha if i == full_edges else 0.0)
                )
                if alpha <= 0.0:
                    continue
                arrow = FancyArrowPatch(
                    geom["start"],
                    geom["end"],
                    arrowstyle="->",
                    mutation_scale=10,
                    linewidth=2.8,
                    color=color,
                    connectionstyle=f"arc3,rad={geom['rad']}",
                    shrinkA=0,
                    shrinkB=0,
                    clip_on=False,
                    alpha=alpha,
                    zorder=4,
                )
                ax.add_patch(arrow)
                active_arrows.append(arrow)

            pulse_list = stage["pulse_nodes"]
            if pulse_list:
                offsets = [positions[name] for name in pulse_list]
                pulse_nodes.set_offsets(offsets)
                pulse_halo.set_offsets(offsets)
                pulse_nodes.set_color([color] * len(offsets))
                pulse_halo.set_color([color] * len(offsets))
                pulse_nodes.set_sizes([200 + 160 * progress] * len(offsets))
                pulse_halo.set_sizes([760 + 520 * progress] * len(offsets))
            else:
                empty_offsets = np.empty((0, 2))
                pulse_nodes.set_offsets(empty_offsets)
                pulse_halo.set_offsets(empty_offsets)
                pulse_nodes.set_sizes([])
                pulse_halo.set_sizes([])

            return [*active_arrows, pulse_nodes, pulse_halo]

        legend_handles = self.flow_legend_handles()
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.04),
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.11)

        animation = FuncAnimation(
            fig,
            update,
            frames=max(len(playback_items) * segment_frames, 1),
            interval=int(1000 / max(fps, 1)),
            blit=True,
            repeat=True,
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output, writer=PillowWriter(fps=fps))
        plt.close(fig)
        return output

    def flow_legend_handles(self) -> list[Patch]:
        return [
            Patch(facecolor="white", edgecolor="#2D2D2D", label="Unobserved"),
            Patch(facecolor="#BDBDBD", edgecolor="#2D2D2D", label="Observed"),
            Patch(facecolor="#2E5BFF", edgecolor="#2D2D2D", label="Treatment"),
            Patch(
                facecolor="none", edgecolor="#2E5BFF", linewidth=2.2, label="Outcome"
            ),
            Patch(
                facecolor=self._FLOW_COLORS["Causal path"],
                edgecolor="none",
                label="Causal path",
            ),
            Patch(
                facecolor=self._FLOW_COLORS["Confounder bias"],
                edgecolor="none",
                label="Confounder bias",
            ),
            Patch(
                facecolor=self._FLOW_COLORS["Mediator bias"],
                edgecolor="none",
                label="Mediator bias",
            ),
            Patch(
                facecolor=self._FLOW_COLORS["Selection bias"],
                edgecolor="none",
                label="Selection bias",
            ),
        ]

    def _draw_flow_base(
        self,
        *,
        ax: Axes,
        observed: set[str],
        treatment_name: str | None,
        outcome_name: str | None,
    ) -> tuple[dict[str, tuple[float, float]], dict[tuple[str, str], _EdgeGeometry]]:
        positions = self._compute_plot_positions()
        node_half_w = 0.45
        node_half_h = 0.25
        edge_gap = 0.06

        edge_radii = self._compute_edge_base_radii(positions)
        edge_geometry: dict[tuple[str, str], _EdgeGeometry] = {}
        for child_name, child in self._variables.items():
            for parent_name in child.parent_names:
                edge = (parent_name, child_name)
                start, end = self._edge_endpoints(
                    positions[parent_name],
                    positions[child_name],
                    node_half_w=node_half_w,
                    node_half_h=node_half_h,
                    edge_gap=edge_gap,
                )
                rad = edge_radii.get(edge, 0.0)
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    mutation_scale=10,
                    linewidth=1.2,
                    color="#B0B4BB",
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=0,
                    shrinkB=0,
                    clip_on=False,
                    zorder=1,
                )
                ax.add_patch(arrow)
                edge_geometry[edge] = {"start": start, "end": end, "rad": rad}

        for name, (x, y) in positions.items():
            face = "#BDBDBD" if name in observed else "white"
            text_color = "#1D1D1F"
            node_edge = "#2D2D2D"
            lw = 1.0
            if treatment_name is not None and name == treatment_name:
                face = "#2E5BFF"
                text_color = "white"
            elif outcome_name is not None and name == outcome_name:
                face = "none"
                node_edge = "#2E5BFF"
                lw = 2.2
                text_color = "#1D1D1F"

            node = FancyBboxPatch(
                (x - node_half_w, y - node_half_h),
                2 * node_half_w,
                2 * node_half_h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=face,
                edgecolor=node_edge,
                linewidth=lw,
                zorder=2,
            )
            node.set_gid(f"node:{name}")
            ax.add_patch(node)
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
                zorder=3,
            )

        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        x_pad = node_half_w + 0.12
        y_pad = node_half_h + 0.12
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        return positions, edge_geometry

    def _validate_observed_names(self, observed: set[str]) -> None:
        unknown = sorted(name for name in observed if name not in self._variables)
        if unknown:
            raise ValueError("Unknown observed variable(s): " + ", ".join(unknown))

    def _compute_edge_base_radii(
        self,
        positions: dict[str, tuple[float, float]],
    ) -> dict[tuple[str, str], float]:
        base: dict[tuple[str, str], float] = {}
        strength = 0.18
        for parent_name in self._variables:
            children = [
                child_name
                for child_name, child in self._variables.items()
                if parent_name in child.parent_names
            ]
            if len(children) <= 1:
                continue
            children = sorted(children, key=lambda c: positions[c][1], reverse=True)
            center = (len(children) - 1) / 2.0
            for idx, child_name in enumerate(children):
                base[(parent_name, child_name)] = base.get(
                    (parent_name, child_name), 0.0
                ) + ((idx - center) * strength)

        for child_name, child in self._variables.items():
            parents = list(child.parent_names)
            if len(parents) <= 1:
                continue
            parents = sorted(parents, key=lambda p: positions[p][1], reverse=True)
            center = (len(parents) - 1) / 2.0
            for idx, parent_name in enumerate(parents):
                base[(parent_name, child_name)] = base.get(
                    (parent_name, child_name), 0.0
                ) - ((idx - center) * (strength * 0.75))
        for edge in list(base):
            base[edge] = float(np.clip(base[edge], -0.36, 0.36))
        return base

    @staticmethod
    def _edge_endpoints(
        parent_xy: tuple[float, float],
        child_xy: tuple[float, float],
        *,
        node_half_w: float,
        node_half_h: float,
        edge_gap: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        x0, y0 = parent_xy
        x1, y1 = child_xy
        dx = x1 - x0
        dy = y1 - y0
        norm = (dx**2 + dy**2) ** 0.5
        if norm <= 1e-9:
            return (x0, y0), (x1, y1)
        ux, uy = dx / norm, dy / norm
        scale = max(
            abs(dx) / max(node_half_w, 1e-9), abs(dy) / max(node_half_h, 1e-9), 1e-9
        )
        off_x = dx / scale
        off_y = dy / scale
        start = (x0 + off_x + ux * edge_gap, y0 + off_y + uy * edge_gap)
        end = (x1 - off_x - ux * edge_gap, y1 - off_y - uy * edge_gap)
        return start, end

    def flow_animation_stages(
        self,
        *,
        treatment_name: str,
        outcome_name: str,
        observed_names: Iterable[str],
    ) -> list[FlowStage]:
        observed = set(observed_names)
        self._validate_treatment_outcome(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=observed,
        )

        all_directed_paths = self._unique_paths(
            [
                path
                for path in self._find_directed_paths(treatment_name, outcome_name)
                if self._is_valid_treatment_outcome_flow(
                    path, treatment_name, outcome_name
                )
            ]
        )
        directed_paths = list(all_directed_paths)

        stages: list[FlowStage] = []
        if directed_paths:
            stages.append(
                {
                    "name": "directed_causal",
                    "label": "Causal path",
                    "color": "#2ECC71",
                    "linestyle": "solid",
                    "pulse_nodes": [],
                    "paths": directed_paths,
                }
            )

        traversal = cast(_TraversalProtocol, self)
        descendants_cache = {
            name: traversal._get_descendants(name) for name in self._variables.keys()
        }
        all_paths = self._find_all_simple_paths(treatment_name, outcome_name)

        confounders = self._latent_confounders(treatment_name, outcome_name, observed)
        for conf in sorted(confounders):
            conf_paths = self._unique_paths(
                [
                    path
                    for path in all_paths
                    if conf in path
                    and self._is_open_path(path, observed, descendants_cache)
                    and self._is_valid_confounder_flow_path(
                        path=path,
                        confounder=conf,
                        outcome_name=outcome_name,
                        observed=observed,
                    )
                    and self._is_valid_treatment_outcome_flow(
                        path, treatment_name, outcome_name
                    )
                ]
            )
            if conf_paths:
                conf_paths = [min(conf_paths, key=lambda p: (len(p), tuple(p)))]
                stages.append(
                    {
                        "name": f"confounder:{conf}",
                        "label": "Confounder bias",
                        "color": "#FF4D4F",
                        "linestyle": "solid",
                        "pulse_nodes": [conf],
                        "paths": conf_paths,
                    }
                )

        mediator_nodes = self._observed_mediators(
            treatment_name, outcome_name, observed
        )
        for mediator in sorted(mediator_nodes):
            mediator_paths = self._unique_paths(
                [path for path in all_directed_paths if mediator in path[1:-1]]
            )
            if mediator_paths:
                stages.append(
                    {
                        "name": f"mediator:{mediator}",
                        "label": "Mediator bias",
                        "color": "#F5A623",
                        "linestyle": "solid",
                        "pulse_nodes": [mediator],
                        "paths": mediator_paths,
                    }
                )

        selection_nodes = self._observed_descendants_of_causal_path(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=observed,
            directed_paths=all_directed_paths,
        )
        if selection_nodes and directed_paths:
            selection_paths = self._unique_paths(directed_paths)
            if selection_paths:
                stages.append(
                    {
                        "name": "selection_bias",
                        "label": "Selection bias",
                        "color": "#9B59B6",
                        "linestyle": "solid",
                        "pulse_nodes": sorted(selection_nodes),
                        "paths": selection_paths,
                    }
                )

        return stages

    @staticmethod
    def _unique_paths(paths: list[list[str]]) -> list[list[str]]:
        seen: set[tuple[str, ...]] = set()
        out: list[list[str]] = []
        for path in paths:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
        return out

    def _validate_treatment_outcome(
        self,
        treatment_name: str,
        outcome_name: str,
        observed: set[str],
    ) -> None:
        if treatment_name not in self._variables:
            raise ValueError(f"Unknown treatment variable '{treatment_name}'.")
        if outcome_name not in self._variables:
            raise ValueError(f"Unknown outcome variable '{outcome_name}'.")
        if treatment_name == outcome_name:
            raise ValueError("`treatment_name` and `outcome_name` must be different.")
        if treatment_name not in observed:
            raise ValueError("`treatment_name` must be included in `observed_names`.")
        if outcome_name in observed:
            raise ValueError("`outcome_name` cannot be included in `observed_names`.")

    def _compute_path_edge_categories(
        self,
        treatment_name: str,
        outcome_name: str,
        conditioned_on: set[str],
    ) -> dict[tuple[str, str], set[str]]:
        edge_categories: dict[tuple[str, str], set[str]] = {}
        flows = self._collect_flow_paths(
            treatment_name=treatment_name,
            outcome_name=outcome_name,
            observed=conditioned_on,
        )
        for category, paths in flows.items():
            for path in paths:
                for idx in range(len(path) - 1):
                    edge = self._path_segment_to_directed_edge(path[idx], path[idx + 1])
                    edge_categories.setdefault(edge, set()).add(category)
        return edge_categories

    def _collect_flow_paths(
        self,
        treatment_name: str,
        outcome_name: str,
        observed: set[str],
    ) -> dict[str, list[list[str]]]:
        descendants_cache = {
            name: cast(_TraversalProtocol, self)._get_descendants(name)
            for name in self._variables.keys()
        }
        flows: dict[str, list[list[str]]] = {
            "open_causal": [],
            "open_noncausal": [],
            "blocked_causal": [],
        }

        for path in self._find_all_simple_paths(treatment_name, outcome_name):
            is_causal = self._is_directed_path(path)
            is_open = self._is_open_path(path, observed, descendants_cache)
            if is_open and is_causal:
                flows["open_causal"].append(path)
            elif is_open and not is_causal:
                flows["open_noncausal"].append(path)
            elif (not is_open) and is_causal:
                flows["blocked_causal"].append(path)

        return flows

    def _find_directed_paths(self, source: str, target: str) -> list[list[str]]:
        children: dict[str, list[str]] = {name: [] for name in self._variables}
        for child_name, variable in self._variables.items():
            for parent_name in variable.parent_names:
                children[parent_name].append(child_name)

        out: list[list[str]] = []

        def dfs(node: str, trail: list[str]) -> None:
            if node == target:
                out.append(list(trail))
                return
            for nxt in children[node]:
                if nxt in trail:
                    continue
                trail.append(nxt)
                dfs(nxt, trail)
                trail.pop()

        dfs(source, [source])
        return out

    def _is_valid_treatment_outcome_flow(
        self,
        path: list[str],
        treatment_name: str,
        outcome_name: str,
    ) -> bool:
        if not path or path[0] != treatment_name or path[-1] != outcome_name:
            return False
        if len(set(path)) != len(path):
            return False
        for node in path[1:-1]:
            if node in {treatment_name, outcome_name}:
                return False
        return True

    def _latent_confounders(
        self,
        treatment_name: str,
        outcome_name: str,
        observed: set[str],
    ) -> set[str]:
        traversal = cast(_TraversalProtocol, self)
        ancestors_t = traversal._get_ancestors(treatment_name)
        ancestors_y = traversal._get_ancestors(outcome_name)
        return {
            name
            for name in ancestors_t.intersection(ancestors_y)
            if name not in observed and name not in {treatment_name, outcome_name}
        }

    def _observed_mediators(
        self,
        treatment_name: str,
        outcome_name: str,
        observed: set[str],
    ) -> set[str]:
        traversal = cast(_TraversalProtocol, self)
        descendants_t = traversal._get_descendants(treatment_name)
        ancestors_y = traversal._get_ancestors(outcome_name)
        return {
            name
            for name in observed
            if name in descendants_t
            and name in ancestors_y
            and name not in {treatment_name, outcome_name}
        }

    def _observed_descendants_of_causal_path(
        self,
        *,
        treatment_name: str,
        outcome_name: str,
        observed: set[str],
        directed_paths: list[list[str]],
    ) -> set[str]:
        path_nodes = {node for path in directed_paths for node in path}
        descendant_nodes: set[str] = set()
        traversal = cast(_TraversalProtocol, self)
        for node in path_nodes:
            descendant_nodes.update(traversal._get_descendants(node))
        descendant_nodes.difference_update(path_nodes)
        descendant_nodes.difference_update({treatment_name, outcome_name})
        return descendant_nodes.intersection(observed)

    def _compute_plot_positions(self) -> dict[str, tuple[float, float]]:
        levels: dict[str, int] = {}
        for name, variable in self._variables.items():
            if variable.parent_names:
                levels[name] = (
                    max(levels[parent] for parent in variable.parent_names) + 1
                )
            else:
                levels[name] = 0

        level_to_names: dict[int, list[str]] = {}
        for name, level in levels.items():
            level_to_names.setdefault(level, []).append(name)

        x_spacing = 2.0
        y_spacing = 1.6
        positions: dict[str, tuple[float, float]] = {}
        for level in sorted(level_to_names):
            names = level_to_names[level]
            n_level = len(names)
            y_offset = (n_level - 1) * y_spacing / 2.0
            for idx, name in enumerate(names):
                positions[name] = (level * x_spacing, y_offset - idx * y_spacing)
        return positions

    @staticmethod
    def _edge_curve_radii(num_paths: int) -> list[float]:
        if num_paths <= 1:
            return [0.0]
        if num_paths == 2:
            return [-0.14, 0.14]
        if num_paths == 3:
            return [-0.2, 0.0, 0.2]
        step = 0.4 / (num_paths - 1)
        return [-0.2 + i * step for i in range(num_paths)]

    def _find_all_simple_paths(self, source: str, target: str) -> list[list[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for child_name, child in self._variables.items():
            for parent_name in child.parent_names:
                adjacency[parent_name].add(child_name)
                adjacency[child_name].add(parent_name)

        paths: list[list[str]] = []

        def dfs(current: str, visited: set[str], trail: list[str]) -> None:
            if current == target:
                paths.append(list(trail))
                return
            for nxt in adjacency[current]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                trail.append(nxt)
                dfs(nxt, visited, trail)
                trail.pop()
                visited.remove(nxt)

        dfs(source, {source}, [source])
        return paths

    def _is_directed_path(self, path: list[str]) -> bool:
        for idx in range(len(path) - 1):
            parent = path[idx]
            child = path[idx + 1]
            if parent not in self._variables[child].parent_names:
                return False
        return True

    def _is_open_path(
        self,
        path: list[str],
        conditioned_on: set[str],
        descendants_cache: dict[str, set[str]],
    ) -> bool:
        if len(path) <= 2:
            return True
        for idx in range(1, len(path) - 1):
            prev_node = path[idx - 1]
            node = path[idx]
            next_node = path[idx + 1]

            prev_points_to_node = prev_node in self._variables[node].parent_names
            next_points_to_node = next_node in self._variables[node].parent_names
            is_collider = prev_points_to_node and next_points_to_node

            if is_collider:
                collider_open = (
                    node in conditioned_on
                    or len(descendants_cache[node].intersection(conditioned_on)) > 0
                )
                if not collider_open:
                    return False
            else:
                if node in conditioned_on:
                    return False
        return True

    def _path_segment_to_directed_edge(
        self, node_a: str, node_b: str
    ) -> tuple[str, str]:
        if node_a in self._variables[node_b].parent_names:
            return (node_a, node_b)
        if node_b in self._variables[node_a].parent_names:
            return (node_b, node_a)
        raise ValueError(
            f"Nodes '{node_a}' and '{node_b}' are not adjacent in the DAG."
        )

    def _is_valid_confounder_flow_path(
        self,
        *,
        path: list[str],
        confounder: str,
        outcome_name: str,
        observed: set[str],
    ) -> bool:
        """Check flow-compatible path for latent confounder bias animation.

        Rules:
        - Path starts at treatment and immediately enters the latent confounder
          through an incoming latent arrow (T <- U).
        - From confounder onward, traversal follows directed outgoing arrows.
        - Intermediate nodes after confounder must be unobserved.
        """
        if len(path) < 3:
            return False
        if path[1] != confounder:
            return False

        treatment = path[0]
        if confounder not in self._variables[treatment].parent_names:
            return False

        for idx in range(1, len(path) - 1):
            current = path[idx]
            nxt = path[idx + 1]
            if current not in self._variables[nxt].parent_names:
                return False
            if nxt != outcome_name and nxt in observed:
                return False
        return True
