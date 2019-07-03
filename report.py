import numpy as np
import torch
import networkx as nx
import igraph

from bokeh.io import save, output_file
from bokeh.models import (
    Plot, Range1d, MultiLine, Circle,
    HoverTool, BoxZoomTool, ResetTool)
from bokeh.models.graphs import from_networkx
from bokeh.io import export_png, export_svgs
from bokeh.transform import linear_cmap
import bokeh.palettes

from joblib import Parallel, delayed
import joblib

from dougu import Results, to_from_idx, mkdir
from dougu.plot import (
    plt, simple_imshow, colors, linestyles, markers, Figure)
from matplotlib import ticker
from ballpark import ballpark

from argparser import get_args
from run_exp import columns, index


def get_graph_data(conf):
    from synthetic_graph import (
        generate_graph, get_graphfile, get_graphdir)
    graphdir = get_graphdir(conf)
    conf_str = graphdir.name
    if conf.graph_sweep:
        graphfile = conf.outdir / 'graph_sweep' / (conf_str + '.graphml')
    else:
        graphfile = get_graphfile(graphdir)
    if not graphfile.exists() or conf.no_cache:
        graph = generate_graph(conf)
        graph.write_graphml(str(graphfile))
    graph = nx.read_graphml(graphfile)
    pagerank = nx.pagerank(graph)

    cachefile = conf.outdir / f'{conf_str}.pkl'
    if cachefile.exists():
        layout = joblib.load(cachefile)
        print(cachefile)
        assert set(layout.keys()) == set(graph.node()), breakpoint()
    else:
        # graph_renderer = from_networkx(
        #     graph, nx.spring_layout, k=0.1, scale=1, center=(0, 0))
        # layout = graph_renderer.layout_provider.to_json(False)['graph_layout']
        import igraph
        ig = igraph.Graph()
        ig.add_vertices(list(graph.nodes))
        # ig.add_edges(list(graph.edges))
        # coords = ig.layout_drl().coords
        print('finding layout')
        coords = ig.layout_fruchterman_reingold().coords
        # coords = ig.layout_kamada_kawai().coords
        node_names = [vertex['name'] for vertex in ig.vs]
        assert len(coords) == len(node_names)
        c = np.array(coords)
        c[:, 0] /= np.abs(c[:, 0]).max()
        c[:, 1] /= np.abs(c[:, 1]).max()
        coords = c.tolist()
        layout = dict(zip(node_names, coords))
        graph_renderer = from_networkx(graph, layout, scale=1, center=(0, 0))
        layout = graph_renderer.layout_provider.to_json(False)['graph_layout']
        joblib.dump(layout, cachefile)
    return {
        'graph': graph,
        'pagerank': pagerank,
        'conf_str': conf_str,
        'graphfile': graphfile,
        'layout': layout}


def plot_graph(conf):
    gd = get_graph_data(conf)
    graph = gd['graph']
    pagerank = gd['pagerank']
    conf_str = gd['conf_str']
    graphfile = gd['graphfile']
    layout = gd['layout']
    if conf.animate:
        pred_files = sorted(conf.rundir.glob('predictions.e*.pt'))
        n_pred_files = len(pred_files)

        def worker(i_and_pred_file):
            i, pred_file = i_and_pred_file
            epoch = int(pred_file.name.split('.')[-2][1:])
            if i < n_pred_files - 1:
                if epoch < 2 or epoch % conf.frame_every_n_epochs != 0:
                    return
            outdir = conf.rundir
            outfile = (outdir / conf_str).with_suffix(f'.e{epoch:06d}.html')
            pred = torch.load(pred_file)
            _plt_graph(
                conf, conf_str, graph, layout, pagerank, outfile, pred=pred)

        if conf.plot_threads > 1:
            tasks = list(enumerate(pred_files))
            Parallel(n_jobs=conf.plot_threads, prefer='threads')(
                delayed(worker)(task) for task in tasks)
        else:
            for i, pred_file in enumerate(pred_files):
                worker((i, pred_file))

        # print(len(set(pred['fw_path'][:, 0].tolist())), 'head entities')
        # print(len(set(pred['fw_target'].tolist())), 'tail entities')
    conf.animate = False
    outdir = mkdir(graphfile.parent / 'fig')
    outfile = (outdir / conf_str).with_suffix('.graph_bokeh.html')
    _plt_graph(conf, conf_str, graph, layout, pagerank, outfile)


def _plt_graph(conf, conf_str, graph, layout, pagerank, outfile, pred=None):
    plot = Plot(
        plot_width=1000,
        plot_height=1000,
        x_range=Range1d(-1.1, 1.1),
        y_range=Range1d(-1.1, 1.1),
        )
    plot.title.text = conf_str
    tooltips = [("index", "@index")]
    if pred is not None:
        edge_colors = {}
        node_vals = {node: [] for node in graph.nodes}
        paths = torch.cat([
            pred['fw_path'], pred['fw_target'].unsqueeze(1)], dim=1)
        for start_node, end_node, edge_data in graph.edges(data=True):
            edge_colors[(start_node, end_node)] = np.nan
        if conf.color_by == 'prob_ratio':
            pred_vals = pred['fw_prob_ratio']
        elif conf.color_by == 'rank':
            ranks = pred['fw_rank']
            pred_vals = 1 / ranks.float()
        else:
            raise ValueError('cannot color nodes by ' + conf.color_by)
        tooltips.append((conf.color_by, "@" + conf.color_by))
        for path, pred_val in zip(paths.tolist(), pred_vals.tolist()):
            start_node, edge_label, end_node = map(str, path)
            edge_colors[(start_node, end_node)] = pred_val
            node_vals[start_node].append(pred_val)
            node_vals[end_node].append(pred_val)
        for node in graph.nodes:
            if not len(node_vals[node]):
                node_vals[node] = np.nan
            else:
                node_vals[node] = np.average(node_vals[node])
        nx.set_edge_attributes(graph, edge_colors, 'edge_color')
        nx.set_node_attributes(graph, node_vals, conf.color_by)

    node_hover_tool = HoverTool(tooltips=tooltips)
    plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())
    graph_renderer = from_networkx(graph, layout)
    s = np.clip(np.power(np.array([
        pagerank[idx]
        for idx in graph_renderer.node_renderer.data_source.data['index']]),
        0.2), .2, 1)
    max_node_size = 10
    pr_sizes = s / s.max() * max_node_size
    if len(graph) > 5000:
        pr_sizes /= 2
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

    if conf.animate:
        node_color = linear_cmap(
            conf.color_by, palette, 0, 1, nan_color='gray')
        line_color = linear_cmap(
            'edge_color', palette, 0, 1, nan_color='gray'),
    else:
        node_color = 'blue'
        line_color = 'blue'

    graph_renderer.node_renderer.glyph = Circle(
        line_color=node_color,
        line_alpha=0.6,
        line_width=0,
        fill_color=node_color,
        fill_alpha=0.9,
        size='pr')
    graph_renderer.edge_renderer.glyph = MultiLine(
        # line_color="edge_color",
        line_color=line_color,
        line_alpha=0.8, line_width=.2)
    plot.renderers = [graph_renderer]
    # if not conf.graph_sweep and not conf.animate:
    # if 'html' in conf.format:
    if True:
        output_file(outfile)
        print(outfile)
        save(plot)
    if 'png' in conf.format:
        png_file = outfile.with_suffix('.png')
        export_png(plot, filename=png_file, timeout=20)
        print(png_file)
    if 'svg' in conf.format:
        plot.output_backend = "svg"
        svg_file = outfile.with_suffix('.svg')
        export_svgs(plot, filename=svg_file)
        print(svg_file)


def plot_results(conf):
    figdir = conf.outdir / 'fig'
    results_store = conf.outdir / conf.exp_name / 'results.h5'

    with Results(results_store, columns, index) as results:
        df = results.df
        if conf.inspect_results:
            breakpoint()
            return
        if conf.emb_dim:
            df = df[df.emb_dim == conf.emb_dim]
        df = df[df.n_layers == conf.n_layers]
        df = df[df.model == conf.model]
        df = df[df.graph_type == conf.graph_type]
        print(len(df), 'data points')
        for val_col in 'epoch', 'acc':
            conf_str = (
                f'{conf.exp_name}.{conf.graph_type}.{conf.model}.'
                f'{val_col}.n_layers_{conf.n_layers}')
            val_df = df[['n_hidden', 'n_paths', val_col]]
            outfile = figdir / f'{conf_str}.matrix.png'
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
            fig_title = f'{conf_str}.lines'
            with Figure(
                    fig_title,
                    xlabel='Number of relation triples',
                    ylabel={'acc': 'Accuracy', 'epoch': 'Epochs'}[val_col],
                    xlim=(0, df.n_paths.max()),
                    ylim=(0, 1.01),
                    figwidth=10,
                    figheight=6):
                print(fig_title)
                for n_hidden, color, linestyle, marker in zip(
                        sorted(df.n_hidden.unique()),
                        colors,
                        linestyles,
                        markers):
                    _df = df[df.n_hidden == n_hidden].sort_values('n_paths')
                    x, y = _df[['n_paths', val_col]].values.T
                    n_params = (
                        (n_hidden + 1) * conf.n_nodes +
                        conf.n_layers * n_hidden)
                    if conf.plot_no_markers:
                        marker = None
                    plt.plot(
                        x, y,
                        color=color, linestyle=linestyle, marker=marker,
                        label=ballpark(n_params))
                ax = plt.gca()
                ax.grid(
                    which='major',
                    linestyle='--',
                    color='gray',
                    alpha=0.3,
                    linewidth=0.8)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
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
