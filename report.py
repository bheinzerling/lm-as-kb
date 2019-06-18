import numpy as np
import torch
import networkx as nx

from bokeh.io import save, output_file
from bokeh.models import (
    Plot, Range1d, MultiLine, Circle,
    HoverTool, BoxZoomTool, ResetTool)
from bokeh.models.graphs import from_networkx
from bokeh.io import export_png, export_svgs
from bokeh.transform import linear_cmap
import bokeh.palettes

from dougu import Results, to_from_idx
from dougu.plot import (
    plt, simple_imshow, colors, linestyles, markers, Figure)
from ballpark import ballpark

from argparser import get_args
from run_exp import columns, index


def plot_graph(conf):
    # from data import KnowledgeGraphPaths
    from synthetic_graph import (
        generate_graph, get_graphfile, get_graphdir)
    graphdir = get_graphdir(conf)
    conf_str = graphdir.name
    if conf.graph_sweep:
        graphfile = conf.outdir / 'graph_sweep' / (conf_str + '.edgelist')
    else:
        graphfile = get_graphfile(graphdir)
    if graphfile.exists():
        graph = nx.read_edgelist(graphfile)
    else:
        graph = generate_graph(conf)
        nx.write_edgelist(graph, graphfile)
    # paths = KnowledgeGraphPaths.load(conf)
    pagerank = nx.pagerank(graph)

    graph_renderer = from_networkx(
        graph, nx.spring_layout, k=0.1, scale=1, center=(0, 0))
    layout = graph_renderer.layout_provider.to_json(False)['graph_layout']
    # TODO: write out memorized edges after every batch/epoch -> one frame
    # loop, change glyph and edge colors in every frame, create animation
    # change glyph color by majority of (not) memorized edges
    # pred_prob as color, red-blue scale
    if conf.animate:
        pred_files = sorted(conf.rundir.glob('predictions.e*.pt'))
        n_pred_files = len(pred_files)
        for i, pred_file in enumerate(pred_files):
            epoch = int(pred_file.name.split('.')[-2][1:])
            if i < n_pred_files - 1:
                if epoch < 2 or epoch % conf.frame_every_n_epochs != 0:
                    continue
            outdir = conf.rundir
            outfile = (outdir / conf_str).with_suffix(f'.e{epoch:06d}.html')
            pred = torch.load(pred_file)
            _plt_graph(
                conf, conf_str, graph, layout, pagerank, outfile, pred=pred)
        print(len(set(pred['fw_path'][:, 0].tolist())), 'head entities')
        print(len(set(pred['fw_target'].tolist())), 'tail entities')
    conf.animate = False
    outdir = graphfile.parent / 'fig'
    outfile = (outdir / conf_str).with_suffix('.graph_bokeh.html')
    _plt_graph(conf, conf_str, graph, layout, pagerank, outfile)


def _plt_graph(conf, conf_str, graph, layout, pagerank, outfile, pred=None):
    plot = Plot(
        plot_width=1000,
        plot_height=1000,
        x_range=Range1d(-1.1, 1.1),
        y_range=Range1d(-1.1, 1.1))
    plot.title.text = conf_str
    tooltips = [("index", "@index")]
    if pred is not None:
        tooltips.append(("prob", "@prob"))
    node_hover_tool = HoverTool(tooltips=tooltips)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    if pred is not None:
        edge_colors = {}
        node_probs = {node: [] for node in graph.nodes}
        paths = torch.cat([
            pred['fw_path'], pred['fw_target'].unsqueeze(1)], dim=1)
        for start_node, end_node, edge_data in graph.edges(data=True):
            edge_colors[(start_node, end_node)] = np.nan
        for path, pred_prob in zip(paths.tolist(), pred['fw_pred'].tolist()):
            start_node, edge_label, end_node = map(str, path)
            edge_colors[(start_node, end_node)] = pred_prob
            node_probs[start_node].append(pred_prob)
            node_probs[end_node].append(pred_prob)
        for node in graph.nodes:
            if not len(node_probs[node]):
                node_probs[node] = np.nan
            else:
                node_probs[node] = np.average(node_probs[node])
        nx.set_edge_attributes(graph, edge_colors, 'edge_color')
        nx.set_node_attributes(graph, node_probs, 'prob')

    graph_renderer = from_networkx(graph, layout)
    s = np.clip(np.power(np.array([
        pagerank[idx]
        for idx in graph_renderer.node_renderer.data_source.data['index']]),
        0.2), .2, 1)
    max_node_size = 10
    pr_sizes = s / s.max() * max_node_size
    graph_renderer.node_renderer.data_source.data['pr'] = pr_sizes

    if conf.cmap.endswith('_r'):
        reverse = True
        cmap_name = conf.cmap[:-2]
    else:
        reverse = False
        cmap_name = conf.cmap
    palettes = getattr(bokeh.palettes, cmap_name)
    palette = palettes[max(palettes.keys())]
    if reverse:
        palette = list(reversed(palette))

    node_color = linear_cmap('prob', palette, 0, 1, nan_color='gray')
    graph_renderer.node_renderer.glyph = Circle(
        line_color=node_color,
        line_alpha=0.6,
        line_width=0,
        fill_color=node_color,
        fill_alpha=0.9,
        size='pr')
    graph_renderer.edge_renderer.glyph = MultiLine(
        # line_color="edge_color",
        line_color=linear_cmap(
            'edge_color', palette, 0, 1, nan_color='gray'),
        line_alpha=0.8, line_width=.2)
    plot.renderers = [graph_renderer]
    # if not conf.graph_sweep and not conf.animate:
    if True:
        output_file(outfile)
        print(outfile)
        save(plot)
    png_file = outfile.with_suffix('.png')
    export_png(plot, filename=png_file)
    print(png_file)
    # plot.output_backend = "svg"
    # svg_file = outfile.with_suffix('.svg')
    # export_svgs(plot, filename=svg_file)
    # print(svg_file)


def plot_results(conf):
    figdir = conf.outdir / 'fig'
    with Results(conf.results_store, columns, index) as results:
        df = results.df
        if conf.emb_dim:
            df = df[df.emb_dim == conf.emb_dim]
        for val_col in 'epoch', 'acc':
            val_df = df[['n_hidden', 'n_paths', val_col]]
            outfile = figdir / f'{conf.exp_name}.{val_col}.matrix.png'
            row_label2idx, row_loc_labels = to_from_idx(
                sorted(df.n_hidden.unique()))
            n_paths = conf.sweep_n_paths or sorted(df.n_paths.unique())
            col_label2idx, col_loc_labels = to_from_idx(n_paths)
            results = np.empty((len(row_loc_labels), len(col_loc_labels)))
            results.fill(np.nan)
            for n_hidden, n_paths, val in val_df.values:
                results[row_label2idx[n_hidden], col_label2idx[n_paths]] = val
            simple_imshow(
                results,
                origin='lower',
                xtick_locs_labels=zip(*col_loc_labels.items()),
                ytick_locs_labels=zip(*row_loc_labels.items()),
                xgrid=np.arange(len(col_loc_labels) - 1) + 0.5,
                ygrid=np.arange(len(row_loc_labels) - 1) + 0.5,
                xlabel='number of relation triples',
                ylabel='model size (hidden units)',
                cmap='cool',
                bad_color='white',
                cbar_title=val_col,
                outfile=outfile)
            print(outfile)

            with Figure(
                    f'{conf.exp_name}.{val_col}.lines',
                    xlabel='Number of relation triples',
                    ylabel={'acc': 'Accuracy', 'epoch': 'Epochs'}[val_col],
                    figwidth=10,
                    figheight=6):
                for n_hidden, color, linestyle, marker in zip(
                        sorted(df.n_hidden.unique()),
                        colors,
                        linestyles,
                        markers):
                    _df = df[df.n_hidden == n_hidden].sort_values('n_paths')
                    x, y = _df[['n_paths', val_col]].values.T
                    n_params = (n_hidden + 1) * conf.n_nodes
                    plt.plot(
                        x, y,
                        color=color, linestyle=linestyle, marker=marker,
                        label=ballpark(n_params))
                ax = plt.gca()
                # box = ax.get_position()
                # Shrink current axis by 20%
                # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(
                    # loc='lower left',
                    bbox_to_anchor=(1.01, 1.0),
                    # ncol=2,
                    borderaxespad=1,
                    # frameon=False,
                    title='Model parameters')


if __name__ == '__main__':
    conf = get_args()
    globals()[conf.command](conf)
