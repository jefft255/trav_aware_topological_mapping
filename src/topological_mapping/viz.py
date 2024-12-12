from dash import Dash, dcc, html, Input, Output, no_update, callback
from topological_mapping.topological_map import (
    TopologicalMap,
    MapNode,
    RealMapNode,
    MapEdge,
    Traversability,
)
import plotly.graph_objects as go
import pandas as pd
import cv2
from typing import List
from threading import Thread
from pathlib import Path

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


def create_bc_marker_from_node(node: MapNode) -> Marker:
    marker = Marker()
    marker.pose.position.x = node.translation[0]
    marker.pose.position.y = node.translation[1]
    marker.pose.position.z = node.translation[2]
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = 1.8
    marker.scale.y = 1.8
    marker.scale.z = 1.8
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.color.r = 0
    marker.color.g = 1
    marker.color.b = 0
    marker.color.a = 1
    marker.header.stamp = rospy.Time.now()
    marker.header.frame_id = "odom"
    marker.id = 0
    return marker


class TopologicalMapRViz:
    def __init__(self, tmap: TopologicalMap):
        self.tmap = tmap
        self.marker_pub = rospy.Publisher(
            "/mapping/viz_marker", MarkerArray, queue_size=1
        )

    @staticmethod
    def create_marker_from_nodes(nodes: List[MapNode]) -> Marker:
        marker_array = MarkerArray()
        for i, node in enumerate(nodes):
            marker = Marker()
            marker.pose.position.x = node.translation[0]
            marker.pose.position.y = node.translation[1]
            marker.pose.position.z = node.translation[2]
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            # Real node red
            if isinstance(node, RealMapNode):
                marker.color.r = 1
                marker.color.g = 0
                marker.color.b = 0
                marker.color.a = 0.7
            # Virtual node green
            else:
                marker.color.r = 0
                marker.color.g = 1
                marker.color.b = 0
                marker.color.a = 0.7
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "odom"
            marker.id = 4 + i
            marker_array.markers.append(marker)
        return marker_array

    @staticmethod
    def create_marker_from_edges(edges: List[MapEdge]) -> Marker:
        colors = [[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]]
        markers = MarkerArray()
        for i, edge_type in enumerate(
            [
                Traversability.DRIVEN,
                Traversability.TRAVERSABLE,
                Traversability.UNTRAVERSABLE,
                Traversability.UNKNOWN,
            ]
        ):
            color = colors[i]
            marker = Marker()
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "odom"
            marker.id = i
            for e in edges:
                if e.traversability is edge_type:
                    for n in [e.nodeA, e.nodeB]:
                        point = Point()
                        point.x = n.translation[0]
                        point.y = n.translation[1]
                        point.z = n.translation[2]
                        marker.points.append(point)
            markers.markers.append(marker)

        return markers

    def refresh(self):
        marker_del = Marker()
        marker_del.action = Marker.DELETEALL
        markers_del = MarkerArray()
        markers_del.markers.append(marker_del)
        self.marker_pub.publish(markers_del)
        rospy.sleep(0.05)

        markers = self.create_marker_from_edges(self.tmap.edges)
        markers.markers += self.create_marker_from_nodes(self.tmap.nodes).markers
        self.marker_pub.publish(markers)


def parse_map(tmap: TopologicalMap):
    xs = [n.translation[0] for n in tmap.nodes]
    ys = [n.translation[1] for n in tmap.nodes]
    zs = [n.translation[2] for n in tmap.nodes]
    # p = Path("/tmp/plotting_tmp")
    # if not p.exists():
    #     p.mkdir()
    # for i, n in enumerate(tmap.nodes):
    #     if n.image_l is None:
    #         continue
    #     img = cv2.hconcat([n.image_l, n.image_f, n.image_r])
    #     p = Path.cwd() / "tmp" / f"{i}.jpg"
    #     cv2.imwrite(str(p), img)

    return xs, ys, zs


def create_map_fig(tmap, traj):
    xs, ys, zs = parse_map(tmap)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    colorscale="viridis",
                    color=zs,
                    size=4,
                    colorbar={"title": "Altitude<br>(m)"},
                    line={"color": "#444"},
                    reversescale=True,
                    sizeref=5,
                    sizemode="diameter",
                    opacity=1.0,
                ),
            ),
            # go.Scatter(
            #     x=[x[0] for x in traj],
            #     y=[x[1] for x in traj],
            #     mode="markers",
            #     marker=dict(
            #         colorscale="viridis",
            #         color=[x[2] for x in traj],
            #         size=2,
            #         colorbar={"title": "Altitude<br>(m)"},
            #         line={"color": "#444"},
            #         reversescale=True,
            #         sizeref=45,
            #         sizemode="diameter",
            #         opacity=0.8,
            #     ),
            # ),
        ],
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    ppi = 96
    height_inches = 3.5
    height_p = height_inches * ppi

    width_inches = 5.0
    width_p = width_inches * ppi

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=width_p,
        height=height_p,
        font_family="Times New Roman",
        title_font_family="Times New Roman",
    )

    draw_axes = False
    if draw_axes:
        for n in tmap.nodes:
            for i, c in enumerate(["red", "yellow", "blue"]):
                fig.add_shape(
                    type="line",
                    x0=n.translation[0],
                    y0=n.translation[1],
                    x1=n.translation[0] + n.rotation[0, i],
                    y1=n.translation[1] + n.rotation[1, i],
                    layer="above",
                    line=dict(
                        color=c,
                    ),
                    xref="x",
                    yref="y",
                )

    d_e = [e for e in tmap.edges if e.traversability is Traversability.DRIVEN]
    t_e = [e for e in tmap.edges if e.traversability is Traversability.TRAVERSABLE]
    u_e = [e for e in tmap.edges if e.traversability is Traversability.UNTRAVERSABLE]
    unk_e = [e for e in tmap.edges if e.traversability is Traversability.UNKNOWN]

    es = unk_e + t_e + u_e + d_e
    for e in es:
        if e.traversability is Traversability.DRIVEN:
            color = "#000000"
            opacity = 1.0
            width = 2.0
        elif e.traversability is Traversability.TRAVERSABLE:
            color = "#0000FF"
            opacity = 0.6
            width = 0.5
        elif e.traversability is Traversability.UNTRAVERSABLE:
            color = "#FF0000"
            opacity = 0.6
            width = 0.5
        elif e.traversability is Traversability.UNKNOWN:
            color = "#F19C99"
            opacity = 0.6
            width = 0.5
        fig.add_shape(
            type="line",
            x0=e.nodeA.translation[0],
            y0=e.nodeA.translation[1],
            x1=e.nodeB.translation[0],
            y1=e.nodeB.translation[1],
            layer="between",
            line=dict(
                color=color,
                width=width,
            ),
            xref="x",
            yref="y",
            opacity=opacity,
        )

    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    fig.update_layout(
        xaxis=dict(title="Easting (m)"),
        yaxis=dict(title="Northing (m)"),
        plot_bgcolor="rgba(255,255,255,0.1)",
    )
    fig.write_image("figB.pdf")
    return fig


def create_map_app(tmap, traj):
    app = Dash("mapviz")

    app.layout = lambda: html.Div(
        [
            dcc.Graph(
                id="graph-basic-2",
                figure=create_map_fig(tmap, traj),
                clear_on_unhover=True,
            ),
            dcc.Tooltip(id="graph-tooltip"),
        ]
    )

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        children = [
            html.Div(
                [
                    # Need to run "python3 -m http.server 8000" in folder /tmp
                    # Annoying but necessary, otherwise need Dash enterprise to use assets
                    html.Img(
                        src=f"http://localhost:8000/{num}.jpg", style={"width": "100%"}
                    ),
                    html.H2(
                        f"Node {num}",
                        style={"color": "darkblue", "overflow-wrap": "break-word"},
                    ),
                    html.P(f"Put node info here"),
                ],
                style={"width": "600px", "white-space": "normal"},
            )
        ]

        return True, bbox, children

    # thread = Thread(target=app.run)
    # thread.start()
    # return thread
    app.run()


if __name__ == "__main__":
    import sys
    import random

    map_path = sys.argv[1]
    map = TopologicalMap.load(Path(map_path))

    # n_nodes = len(map.nodes)
    # n_to_keep = 150
    # n_to_lose = n_nodes - n_to_keep

    # id_to_remove = random.sample(range(n_nodes), n_to_lose)
    # nodes_to_remove = [map.nodes[i] for i in id_to_remove]

    # for n in nodes_to_remove:
    #     map.remove_node(n)
    create_map_app(map, None)
